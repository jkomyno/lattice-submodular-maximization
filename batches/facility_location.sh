#!/bin/sh

# ml Python/3.8.6-GCCcore-10.2.0

#SBATCH --time=24:00:00
/bin/hostname

# Facility Location (our algorithms)
srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=facility_location \
  algo=SGL-I

srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=facility_location \
  algo=SGL-II

srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=facility_location \
  algo=SGN-III

# Facility Location (other algorithms)
srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=facility_location \
  algo=Soma-DR-I

srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=facility_location \
  algo=SSG

# sbatch
#    -n, --ntasks=ntasks         number of tasks to run
#        --ntasks-per-node=n     number of tasks to invoke on each node
#    -N, --nodes=N               number of nodes on which to run (N = min[-max])
#    -o, --output=out            file for batch script's standard output
#
# sbatch --ntasks=5 --nodes=5 -o stdout.facility_location.txt -e stderr.facility_location.txt ./batches/facility_location.sh
