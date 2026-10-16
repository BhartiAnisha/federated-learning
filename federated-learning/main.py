"""Entry point.  Examples:
    python main.py --quick --datasets hospital --results-dir /tmp/smoke      # smoke test
    python main.py --datasets hospital bank iot --stages all                # full experiment
    python -m experiments.report                                            # tables + figures
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from experiments.run_experiments import main

if __name__ == "__main__":
    main()
