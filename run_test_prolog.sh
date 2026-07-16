#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
scryer-prolog -f --no-add-history main.pl -g 'run_tests,halt'