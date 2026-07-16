# fuzzy_match FFI

This directory exposes the Python `fuzzy_score(a, b)` and
`fuzzy_match(a, b, threshold)` functions through a C ABI shared library. The
library embeds CPython, imports `main.py` from the shared library's directory,
and keeps the model loaded between calls.

## Build

Install the Python dependencies first, then build the shared library:

```sh
python3 -m pip install -r fuzzy_match/requirements.txt
make -C fuzzy_match
```

`python3-config` must describe the same Python installation containing the
dependencies. By default, the Makefile uses `../.venv/bin/python` when it
exists and otherwise uses `python3` to locate installed packages. Override the
interpreter or build configuration when necessary, for example:

```sh
make -C fuzzy_match PYTHON=/path/to/python PYTHON_CONFIG=/path/to/python3-config
```

## Use from Scryer Prolog

Run Scryer from the repository root so the relative library path in `main.pl`
resolves correctly:

```sh
scryer-prolog main.pl
```

Example queries:

```prolog
?- fuzzy_match("eosinophilic", "pink", 0.3).
	true.
?- fuzzy_score("eosinophilic", "pink", Score).
	Score = 0.999...
?- run_tests.
	All fuzzy-match FFI cases passed.
	true.
```

The native exports have two and three input parameters respectively. Scryer's
FFI maps a non-boolean C return value to one additional Prolog output argument,
so the generated and wrapped score predicate is `fuzzy_score/3`; the boolean
match predicate remains `fuzzy_match/3`.