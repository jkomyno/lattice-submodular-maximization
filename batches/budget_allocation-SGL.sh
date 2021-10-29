#!/bin/bash

#SBATCH -J budget_allocation-SGL     # job name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=4          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=32GB                   # memory per node
#SBATCH --time=4:00:00               # time limit
#SBATCH --error=stderr.budget_allocation.SGL.txt            
#SBATCH --output=stdout.budget_allocation.SGL.txt           
#SBATCH --partition=cpufast          # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-III

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-III-b

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-III-c

python -u main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-II-b
