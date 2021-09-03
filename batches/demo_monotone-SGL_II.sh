#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=3GB                    # memory per node
#SBATCH --time=4:00:00               # time limit
#SBATCH --error=stderr.demo_monotone.SGL_II.txt            
#SBATCH --output=stdout.demo_monotone.SGL_II.txt           
#SBATCH --partition=cpu              # partition name
ml Python/3.8.6-GCCcore-10.2.0

python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=SGL-II
