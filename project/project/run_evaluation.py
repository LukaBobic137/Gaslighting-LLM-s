
import sys
import os

import pysqlite3.dbapi2 as sqlite3
sys.modules['sqlite3'] = sqlite3

sys.path.insert(0, '/scratch/mhlupic/semeval_data/semeval25-unlearning-data')

if len(sys.argv) == 1:
    print("This wrapper forwards arguments to the evaluation script.")
    print("Required: --data_path and --checkpoint_path (unless using --compute_metrics_only).")
    print("")
    print("Example:")
    print("  python run_evaluation.py --data_path /scratch/mhlupic/train/ --checkpoint_path GOLD_output-unlearned-7B")
    sys.exit(1)

from evaluate_generations import main

if __name__ == '__main__':
    main()
