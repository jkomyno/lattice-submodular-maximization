#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1          # tasks per node
#SBATCH --cpus-per-task=1            # cpus per task
#SBATCH --mem=8GB                    # memory per node
#SBATCH --time=48:00:00              # time limit
#SBATCH --error=stderr.facility_location.Soma_II.txt            
#SBATCH --output=stdout.facility_location.Soma_II.txt           
#SBATCH --partition=cpulong          # partition name
ml Python/3.8.6-GCCcore-10.2.0

python -u main.py \
  -m runtime=cluster \
  obj=facility_location \
  algo=Soma-II
