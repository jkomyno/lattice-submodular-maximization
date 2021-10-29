#!/bin/bash

# cpufast
sbatch ./batches/budget_allocation-SGL.sh
sbatch ./batches/demo_monotone-SGL.sh
sbatch ./batches/demo_monotone_skewed-SGL.sh
sbatch ./batches/demo_non_monotone-SGL.sh

# cpulong
sbatch ./batches/budget_allocation-base.sh
sbatch ./batches/demo_monotone-base.sh
sbatch ./batches/demo_monotone_skewed-base.sh
sbatch ./batches/demo_non_monotone-base.sh

