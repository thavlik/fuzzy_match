PYTHON_CONFIG ?= python3-config
PYTHON ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python3)
CC ?= cc

CPPFLAGS += $(shell $(PYTHON_CONFIG) --includes)
PYTHON_SITE_PACKAGES := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
PYTHON_LIBRARY := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY"))')
CPPFLAGS += -DFUZZY_PYTHON_SITE_PACKAGES='"$(PYTHON_SITE_PACKAGES)"'
CPPFLAGS += -DFUZZY_PYTHON_LIBRARY='"$(PYTHON_LIBRARY)"'
CFLAGS ?= -O2
CFLAGS += -std=c11 -fPIC -Wall -Wextra -Wpedantic
LDFLAGS += -shared
LDLIBS += $(shell $(PYTHON_CONFIG) --embed --ldflags) -pthread

.PHONY: all clean

all: libfuzzy_match.so

libfuzzy_match.so: ffi_bridge.c ffi_bridge.h main.py
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ ffi_bridge.c $(LDLIBS)

clean:
	rm -f libfuzzy_match.so