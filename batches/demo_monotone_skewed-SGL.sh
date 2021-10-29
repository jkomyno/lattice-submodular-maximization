#!/bin/bash

#SBATCH -J demo_monotone_skewed-SGL  # job name
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=4          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=32GB                   # memory per node
#SBATCH --time=4:00:00               # time limit
#SBATCH --error=stderr.demo_monotone_skewed.SGL.txt            
#SBATCH --output=stdout.demo_monotone_skewed.SGL.txt           
#SBATCH --partition=cpufast          # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u main.py \
  -m runtime=cluster \
  obj=demo_monotone_skewed \
  algo=SGL-III

python -u main.py \
  -m runtime=cluster \
  obj=demo_monotone_skewed \
  algo=SGL-III-b

python -u main.py \
  -m runtime=cluster \
  obj=demo_monotone_skewed \
  algo=SGL-III-c

python -u main.py \
  -m runtime=cluster \
  obj=demo_monotone_skewed \
  algo=SGL-II-b
