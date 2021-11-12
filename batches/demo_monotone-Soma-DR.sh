#!/bin/bash

#SBATCH -J demo_monotone-Soma-DR     # job name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=32GB                   # memory per node
#SBATCH --time=24:00:00              # time limit
#SBATCH --error=stderr.demo_monotone-Soma-DR.txt            
#SBATCH --output=stdout.demo_monotone-Soma-DR.txt           
#SBATCH --partition=cpu              # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u -m python.benchmark  \
  runtime=cluster \
  obj=demo_monotone \
  algo=Soma-DR
