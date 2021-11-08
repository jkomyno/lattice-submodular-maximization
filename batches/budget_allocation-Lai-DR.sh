#!/bin/bash

#SBATCH -J budget_allocation-Lai-DR  # job name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=32GB                   # memory per node
#SBATCH --time=24:00:00              # time limit
#SBATCH --error=stderr.budget_allocation-Lai-DR.txt            
#SBATCH --output=stdout.budget_allocation-Lai-DR.txt           
#SBATCH --partition=cpu              # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u -m python.benchmark  \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=Lai-DR
