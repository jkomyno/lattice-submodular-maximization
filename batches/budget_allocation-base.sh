#!/bin/bash

#SBATCH -J budget_allocation-base    # job name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=3          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=32GB                   # memory per node
#SBATCH --time=3:00:00:00            # time limit
#SBATCH --error=stderr.budget_allocation.base.txt            
#SBATCH --output=stdout.budget_allocation.base.txt           
#SBATCH --partition=cpulong          # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=Soma-DR-I

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SSG

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=Soma-DR-II
