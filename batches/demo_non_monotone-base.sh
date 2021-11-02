#!/bin/bash

#SBATCH -J demo_non_monotone-base    # job name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=3          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=32GB                   # memory per node
#SBATCH --time=72:00:00              # time limit
#SBATCH --error=stderr.demo_non_monotone.base.txt            
#SBATCH --output=stdout.demo_non_monotone.base.txt           
#SBATCH --partition=cpulong          # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u -m python.benchmark  \
  -m runtime=cluster \
  obj=demo_non_monotone \
  algo=Soma-DR-I

python -u -m python.benchmark  \
  -m runtime=cluster \
  obj=demo_non_monotone \
  algo=SSG

python -u -m python.benchmark  \
  -m runtime=cluster \
  obj=demo_non_monotone \
  algo=Soma-II