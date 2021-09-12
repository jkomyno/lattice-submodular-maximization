#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=16GB                   # memory per node
#SBATCH --time=4:00:00               # time limit
#SBATCH --error=stderr.budget_allocation.SGL_III_b.txt            
#SBATCH --output=stdout.budget_allocation.SGL_III_b.txt           
#SBATCH --partition=cpulong          # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-III-b
