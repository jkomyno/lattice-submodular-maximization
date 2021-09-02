#!/bin/sh

# ml Python/3.8.6-GCCcore-10.2.0

#SBATCH --time=24:00:00
/bin/hostname

# Budget Allocation (our algorithms)
srun -p cpulong \
  --mincpus=1 \
  --mem=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-I

srun -p cpulong \
  --mincpus=1 \
  --mem=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGL-II

srun -p cpulong \
  --mincpus=1 \
  --mem=6000 \
  -l python main.py \
  -m runtime=cluster \
  obj=budget_allocation \
  algo=SGN-III

# sbatch
#    -n, --ntasks=ntasks         number of tasks to run
#        --ntasks-per-node=n     number of tasks to invoke on each node
#    -N, --nodes=N               number of nodes on which to run (N = min[-max])
#    -o, --output=out            file for batch script's standard output
#
# sbatch --ntasks=3 --nodes=3 -o stdout.budget_allocation.txt -e stderr.budget_allocation.txt ./batches/budget_allocation.sh
