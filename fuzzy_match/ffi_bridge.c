#define _GNU_SOURCE

#include "ffi_bridge.h"

#include <Python.h>
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

static pthread_once_t python_once = PTHREAD_ONCE_INIT;
static PyObject *python_fuzzy_score = NULL;
static PyObject *python_fuzzy_match = NULL;
static const char library_anchor = 0;

static int prepend_python_path(const char *directory)
{
    PyObject *path = PySys_GetObject("path");
    PyObject *python_directory = PyUnicode_DecodeFSDefault(directory);
    if (path == NULL || python_directory == NULL) {
        Py_XDECREF(python_directory);
        return -1;
    }

    int result = PyList_Insert(path, 0, python_directory);
    Py_DECREF(python_directory);
    return result;
}

static int prepend_library_directory_to_python_path(void)
{
    Dl_info info;
    if (dladdr(&library_anchor, &info) == 0 || info.dli_fname == NULL) {
        fprintf(stderr, "fuzzy_match FFI: unable to locate the shared library\n");
        return -1;
    }

    char directory[4096];
    size_t length = strlen(info.dli_fname);
    if (length >= sizeof(directory)) {
        fprintf(stderr, "fuzzy_match FFI: shared library path is too long\n");
        return -1;
    }

    memcpy(directory, info.dli_fname, length + 1);
    char *separator = strrchr(directory, '/');
    if (separator == NULL) {
        memcpy(directory, ".", 2);
    } else if (separator == directory) {
        separator[1] = '\0';
    } else {
        *separator = '\0';
    }

    return prepend_python_path(directory);
}

static void initialize_python(void)
{
#ifdef FUZZY_PYTHON_LIBRARY
    /* Scryer opens foreign libraries locally. CPython extension modules expect
       the interpreter API in the global symbol table, so promote libpython
       before importing any native Python packages. */
    if (dlopen(FUZZY_PYTHON_LIBRARY, RTLD_NOW | RTLD_GLOBAL) == NULL) {
        fprintf(stderr, "fuzzy_match FFI: failed to promote %s: %s\n",
                FUZZY_PYTHON_LIBRARY, dlerror());
    }
#endif

    Py_Initialize();

#ifdef FUZZY_PYTHON_SITE_PACKAGES
    if (prepend_python_path(FUZZY_PYTHON_SITE_PACKAGES) != 0) {
        fprintf(stderr, "fuzzy_match FFI: failed to add Python site-packages\n");
    }
#endif

    if (prepend_library_directory_to_python_path() == 0) {
        PyObject *module = PyImport_ImportModule("main");
        if (module != NULL) {
            python_fuzzy_score = PyObject_GetAttrString(module, "fuzzy_score");
            python_fuzzy_match = PyObject_GetAttrString(module, "fuzzy_match");
            Py_DECREF(module);

            if (python_fuzzy_score != NULL && !PyCallable_Check(python_fuzzy_score)) {
                Py_CLEAR(python_fuzzy_score);
            }
            if (python_fuzzy_match != NULL && !PyCallable_Check(python_fuzzy_match)) {
                Py_CLEAR(python_fuzzy_match);
            }
        }
    }

    if (python_fuzzy_score == NULL || python_fuzzy_match == NULL) {
        fprintf(stderr, "fuzzy_match FFI: failed to import Python callables\n");
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
    }

    /* The process owns the interpreter for its lifetime. Release the GIL so
       every foreign call can acquire it safely, regardless of its thread. */
    PyEval_SaveThread();
}

static PyObject *python_string(const char *value)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_ValueError, "fuzzy_match received a null string");
        return NULL;
    }

    return PyUnicode_DecodeUTF8(value, (Py_ssize_t)strlen(value), "strict");
}

FUZZY_API double fuzzy_score(const char *a, const char *b)
{
    pthread_once(&python_once, initialize_python);
    PyGILState_STATE gil = PyGILState_Ensure();
    double score = NAN;

    if (python_fuzzy_score != NULL) {
        PyObject *python_a = python_string(a);
        PyObject *python_b = python_string(b);
        if (python_a != NULL && python_b != NULL) {
            PyObject *result = PyObject_CallFunctionObjArgs(
                python_fuzzy_score, python_a, python_b, NULL);
            if (result != NULL) {
                score = PyFloat_AsDouble(result);
                Py_DECREF(result);
            }
        }
        Py_XDECREF(python_a);
        Py_XDECREF(python_b);
    }

    if (PyErr_Occurred()) {
        fprintf(stderr, "fuzzy_match FFI: fuzzy_score failed\n");
        PyErr_Print();
        score = NAN;
    }

    PyGILState_Release(gil);
    return score;
}

FUZZY_API uint8_t fuzzy_match(const char *a, const char *b, double threshold)
{
    pthread_once(&python_once, initialize_python);
    PyGILState_STATE gil = PyGILState_Ensure();
    uint8_t matches = 0;

    if (python_fuzzy_match != NULL) {
        PyObject *python_a = python_string(a);
        PyObject *python_b = python_string(b);
        PyObject *python_threshold = PyFloat_FromDouble(threshold);
        if (python_a != NULL && python_b != NULL && python_threshold != NULL) {
            PyObject *result = PyObject_CallFunctionObjArgs(
                python_fuzzy_match, python_a, python_b, python_threshold, NULL);
            if (result != NULL) {
                int truth = PyObject_IsTrue(result);
                if (truth >= 0) {
                    matches = (uint8_t)truth;
                }
                Py_DECREF(result);
            }
        }
        Py_XDECREF(python_a);
        Py_XDECREF(python_b);
        Py_XDECREF(python_threshold);
    }

    if (PyErr_Occurred()) {
        fprintf(stderr, "fuzzy_match FFI: fuzzy_match failed\n");
        PyErr_Print();
        matches = 0;
    }

    PyGILState_Release(gil);
    return matches;
}