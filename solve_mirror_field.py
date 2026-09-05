"""Run with: conda activate tf2; python solve_mirror_field.py --help.

All problem/ files are read-only inputs. See README.md for stages and budgets.
"""
import os

# Parallelism is across independent designs/scrambles; avoid nested BLAS pools.
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')

from heliostat.cli import main

if __name__=='__main__':
    main()
