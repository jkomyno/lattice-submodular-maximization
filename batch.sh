#!/bin/bash

# squeue --format="%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R" --me

sbatch ./batches/demo_monotone-SGL-III.sh
sbatch ./batches/demo_monotone-Lai-DR.sh
sbatch ./batches/demo_monotone-SSG.sh
sbatch ./batches/demo_monotone-Soma-DR.sh
