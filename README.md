# Fuzzy matching for Scryer Prolog

This repository exposes model-backed fuzzy matching to
[Scryer Prolog](https://www.scryer.pl/) through its native C foreign-function
interface (FFI). The implementation uses Meta's
`meta-llama/Llama-3.1-8B-Instruct` model to decide whether two medical or
scientific expressions are synonyms or accurate descriptions of one another.

Two native functions are exported by `libfuzzy_match.so`:

- `fuzzy_match(const char *concept, const char *target, double threshold)`
- `fuzzy_score(const char *concept, const char *target)`

They are available from Prolog as:

- `fuzzy_match(+Concept, +Target, +Threshold)` — succeeds when the score is
  greater than or equal to `Threshold`.
- `fuzzy_score(+Concept, +Target, -Score)` — unifies `Score` with a floating-point
  value between 0 and 1.

> Scryer adds an output argument for non-boolean native return values. The C
> `fuzzy_score` export therefore has two input parameters, while its Prolog
> predicate is `fuzzy_score/3`.

## How scoring works

Scoring uses a hierarchical classifier built from one or two few-shot Llama
prompts. Each stage reads the logits at the first answer token and normalizes
them over that stage's allowed single-token labels.

The equivalence stage always runs and distinguishes:

- `A` — equivalent: synonyms, colloquial descriptions, morphology, stain
  colors, clinical shorthand, or functional definitions of the same idea.
- `C` — unrelated: medically associated expressions that do not describe the
  same idea.

Its conditional equivalence probability is:

$$
P_E = \frac{\exp(\ell_A)}{\exp(\ell_A) + \exp(\ell_C)}
$$

where $\ell_A$ and $\ell_C$ are the first-token logits for labels `A` and `C`.

Some targets contain assertion or directional cues such as `no`, `without`,
`absent`, `normal`, `reduces`, or `as needed`. For these pairs, a separate
contradiction stage runs first and distinguishes:

- `A` — not contradictory: equivalent, compatible, or merely unrelated.
- `B` — contradictory: opposite or mutually incompatible clinical states.

The contradiction probability is computed from its `A` and `B` logits in the
same way. If $P(\text{contradictory}) \ge 0.5$, contradiction vetoes semantic
similarity and the final score is `0.0`. This prevents pairs such as
`epistaxis` / `no nasal bleeding` from matching solely because their medical
content overlaps. The cue detector treats assertional phrases such as
`negative for` separately from terminology such as `gram-negative`.

If the contradiction veto does not fire, the result is:

$$
\operatorname{score} =
\begin{cases}
P_E, & P_E > 0.5 \\
0, & \text{otherwise}
\end{cases}
$$

`fuzzy_match` succeeds when this final score is at least the caller-supplied
threshold. Consequently, an unrelated or contradictory pair receives exactly
zero rather than a low nonzero similarity score. Since equivalence must beat
unrelatedness before a nonzero score is returned, nonzero scores are greater
than `0.5`.

The prompts are calibrated for medical and scientific vocabulary, including
plain-language descriptions, histology, morphology, microbiology, symptoms,
and treatments. Scores outside this domain have not been calibrated. Ordinary
pairs require one model inference; pairs that activate contradiction detection
require two.

## Architecture

```mermaid
flowchart LR
    P[Scryer Prolog] -->|cstr, double| F[Scryer FFI / libffi]
    F --> C[libfuzzy_match.so]
    C -->|CPython C API| Y[fuzzy_match/main.py]
    Y --> T[Transformers + PyTorch]
    T --> M[Llama 3.1 8B Instruct on CUDA]
```

The native bridge:

1. Initializes an embedded CPython interpreter once per Scryer process.
2. Adds the selected Python environment's `site-packages` and the shared
   library directory to Python's import path.
3. Imports and caches `fuzzy_score` and `fuzzy_match` from
   `fuzzy_match/main.py`.
4. Acquires the Python GIL around every foreign call.
5. Converts Scryer strings to UTF-8 Python strings and converts Python results
   back to C values.

The model is loaded during the first FFI call and remains resident for the
lifetime of the process. Python exceptions and tracebacks are printed to
standard error; native failures return `NaN` from `fuzzy_score` or false from
`fuzzy_match`.

## Requirements

### Runtime

- Linux (the current bridge builds an ELF `.so` and uses `dlopen`/`pthread`)
- A CUDA-capable NVIDIA GPU with bfloat16 support
- Enough GPU memory for Llama 3.1 8B in bfloat16 (approximately 16 GB for model
  weights, with additional runtime overhead)
- Python 3 with the dependencies in `fuzzy_match/requirements.txt`
- Scryer Prolog with `library(ffi)` support

### Build tools

- A C11 compiler such as GCC or Clang
- GNU Make
- Python development headers and the embeddable Python library
- `python3-config` with support for `--embed`

On Debian or Ubuntu, the native prerequisites are typically provided by:

```sh
sudo apt install build-essential python3-dev python3-venv
```

Install Scryer Prolog separately according to its upstream documentation. The
`scryer-prolog` executable must be on `PATH`.

### Model access

`meta-llama/Llama-3.1-8B-Instruct` is a gated Hugging Face model. Before the
first run:

1. Request or accept access to the model on Hugging Face.
2. Authenticate locally with a token that has permission to download it.

For example, after installing `huggingface_hub`:

```sh
hf auth login
```

The first invocation downloads the model into the Hugging Face cache and may
take significant time and disk space. Later processes reuse that cache, but
each new Scryer process loads the model into GPU memory again.

## Setup and build

Run all commands from the repository root.

Create a virtual environment and install the Python dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r fuzzy_match/requirements.txt
```

Build the shared library:

```sh
make -C fuzzy_match
```

The result is `fuzzy_match/libfuzzy_match.so`. It is a generated file and is
ignored by Git.

The Makefile uses `../.venv/bin/python` when that environment exists; otherwise
it uses `python3`. It uses `python3-config` for compiler and linker flags. If
these refer to different Python installations, or if a different environment
is desired, override both values explicitly:

```sh
make -C fuzzy_match \
  PYTHON=/path/to/python \
  PYTHON_CONFIG=/path/to/python3-config
```

The Python interpreter and `python3-config` must use a compatible Python ABI.
To rebuild from scratch:

```sh
make -C fuzzy_match clean all
```

## Usage

Because `main.pl` loads `./fuzzy_match/libfuzzy_match.so` by relative path,
start Scryer from the repository root:

```sh
scryer-prolog main.pl
```

### Boolean matching

```prolog
?- fuzzy_match("eosinophilic", "pink", 0.5).
   true.

?- fuzzy_match("basophilic", "red", 0.5).
   false.
```

`Concept` and `Target` must be Scryer strings (lists of characters), as shown
with double quotes. `Threshold` is a floating-point input.

### Retrieving a score

```prolog
?- fuzzy_score("eosinophilic", "pink", Score).
  Score = 0.97....
```

The exact score can vary slightly with library versions, GPU hardware, and
floating-point behavior.

### Running the test suite interactively

```prolog
?- run_tests.
Should pass (min=0.9..., max=0.9...):
Concept: comma-shaped    | Target: curved rod    | Spread Score: 0.9...
...

Should fail (min=0.0000, max=0.0000):
Concept: basophilic      | Target: red           | Spread Score: 0.0000
...

All fuzzy-match FFI cases passed.
   true.
```

## Tests

The test threshold is `0.5`. There are 13 expected matches and 4 expected
non-matches covering stain colors, morphology, muscle tone, organisms, and
clinical descriptions. The canonical Python case list is in
`fuzzy_match/tests/cases.py`; equivalent Prolog facts are maintained in
`main.pl` and should be kept synchronized.

Run the complete Prolog test harness with:

```sh
./run_test_prolog.sh
```

or directly:

```sh
scryer-prolog -f --no-add-history main.pl -g 'run_tests,halt'
```

The harness:

1. Calls `fuzzy_score/3` for every case.
2. Prints every score plus minimum and maximum statistics for each group.
3. Calls `fuzzy_match/3` for every case to test the boolean FFI path.
4. Fails and prints the concept, target, score, and threshold if an expectation
   is violated.

Because both exported paths are tested, each case is scored twice. Each score
uses one model inference for ordinary pairs or two when contradiction cues are
present.

The Python implementation can also print its score report directly:

```sh
.venv/bin/python fuzzy_match/main.py
```

## Repository layout

```text
.
├── main.pl                         Scryer bindings, examples, and test harness
├── run_test_prolog.sh              Non-interactive Prolog test runner
├── fuzzy_match/
│   ├── main.py                     Model loading and scoring implementation
│   ├── ffi_bridge.c                Embedded-CPython C ABI bridge
│   ├── ffi_bridge.h                Exported native declarations
│   ├── Makefile                    Shared-library build
│   ├── requirements.txt            Python dependencies
│   └── tests/
│       ├── cases.py                Canonical passing and failing pairs
│       └── compare_*.py            Experimental model comparisons
└── notes.txt                       Original implementation requirements
```

The comparison scripts under `fuzzy_match/tests/` explore alternative language
models, rerankers, and biomedical embedding models. They are experiments rather
than part of the Prolog FFI runtime and may download additional models or need
more resources.

## Development notes

- Keep the facts in `main.pl` synchronized with
  `fuzzy_match/tests/cases.py` whenever cases change.
- Rebuild `libfuzzy_match.so` after changing the C bridge. The Makefile also
  rebuilds it when `fuzzy_match/main.py` changes.
- Run Scryer from the repository root unless the library path in `main.pl` is
  changed.
- The interpreter is intentionally not finalized: it and the loaded model are
  owned for the lifetime of the Scryer process.
- The bridge manages the GIL for each call, but GPU/model concurrency behavior
  is governed by PyTorch and Transformers.
