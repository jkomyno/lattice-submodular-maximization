#!/bin/sh

# ml Python/3.8.6-GCCcore-10.2.0

#SBATCH --time=24:00:00
/bin/hostname

# Demo Monotone (our algorithms)
srun -p cpu \
  --mincpus=1 \
  --mem-per-cpu=2000 \
  -l python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=SGL-I

srun -p cpu \
  --mincpus=1 \
  --mem-per-cpu=2000 \
  -l python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=SGL-II

srun -p cpu \
  --mincpus=1 \
  --mem-per-cpu=2000 \
  -l python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=SGN-III

# Demo Monotone (other algorithms)

srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=2000 \
  -l python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=Soma-DR-I

srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=2000 \
  -l python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=SSG

srun -p cpulong \
  --mincpus=1 \
  --mem-per-cpu=2000 \
  -l python main.py \
  -m runtime=cluster \
  obj=demo_monotone \
  algo=Soma-II

# sbatch
#    -n, --ntasks=ntasks         number of tasks to run
#        --ntasks-per-node=n     number of tasks to invoke on each node
#    -N, --nodes=N               number of nodes on which to run (N = min[-max])
#    -o, --output=out            file for batch script's standard output
#
# sbatch --ntasks=6 --nodes=6 -o stdout.demo_monotone.txt -e stderr.demo_monotone.txt ./batches/demo_monotone.sh
