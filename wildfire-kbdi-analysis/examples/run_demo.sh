#!/usr/bin/env bash
set -euo pipefail
python examples/make_synthetic_data.py --output examples/_synthetic_input
python src/kbdi/run_cmip6_kbdi.py --input-root examples/_synthetic_input --output-root examples/_synthetic_results --historical-start 2000 --historical-end 2001 --baseline-start 2000 --baseline-end 2001 --start-year 2002 --end-year 2002 --models MODEL-A MODEL-B --scenarios ssp126 ssp245
python src/kbdi/ensemble.py --model-output-root examples/_synthetic_results --output-root examples/_synthetic_mme --start-year 2002 --end-year 2002 --models MODEL-A MODEL-B --scenarios ssp126 ssp245 --min-models 2
