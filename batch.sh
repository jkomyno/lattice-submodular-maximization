#!/bin/bash

# squeue --format="%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R" --me

# cpufast
sbatch ./batches/budget_allocation-SGL.sh
sbatch ./batches/demo_monotone-SGL.sh
sbatch ./batches/demo_monotone_skewed-SGL.sh
# sbatch ./batches/demo_non_monotone-SGL.sh

# cpulong
sbatch ./batches/budget_allocation-base.sh
sbatch ./batches/demo_monotone-base.sh
sbatch ./batches/demo_monotone_skewed-base.sh
# sbatch ./batches/budget_allocation-Soma-II.sh
# sbatch ./batches/demo_monotone-Soma-II.sh
# sbatch ./batches/demo_monotone_skewed-Soma-II.sh
# sbatch ./batches/demo_non_monotone-base.sh

