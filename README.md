# Randomized Maximization of Monotone DR-Submodular Functions on the Integer Lattice
## Tech stack

- Python 3.8 or superior
- [Hydra](https://hydra.cc) configuration framework

## Initialize project

Install the virtual environment:

```bash
python3 -m venv ./venv
```

Activate the virtual environment:

```bash
source ./venv/bin/activate
```

Install third-party dependencies:

```bash
python3 -m pip install -r python/requirements.txt
```

## How to run

This project is composed of 2 Python modules (one for each step of the application) that should be run in sequence.
The output of the application is stored in the [/out](/out) folder in a hierarchical fashion.

### (1) Benchmark Step: python.benchmark

- Compute the ground truth density map (which associates each subset to the probability of being sampled from the ground truth distribution) only once
- Run the specified samplers over the target probabilistic submodular models, saving the history of the samples obtained in CSV

```bash
python3 -u -m python.benchmark -m \
  obj=demo_monotone_skewed \
  runtime=laptop \
  algo=SSG,SGL-III-b,SGL-III-c,Soma-DR-I
```

### (2) Plot generation Step: python.plotter

- Plot the cumulative probability distance between each sampler's outcome and the ground truth distribution

```bash
python3 -u -m python.plotter
```

## Different types of benchmark experiments:

#### Run experiments on the Synthetic DR-Monotone Submodular Function

```bash
python3 -u -m python.benchmark -m obj=demo_monotone \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Run experiments on the Budget Allocation Problem

```bash
python3 -u -m python.benchmark -m obj=budget_allocation \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Run experiments on the Facility Location Problem

```bash
python3 -u -m python.benchmark -m obj=facility_location \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Run experiments on the Synthetic Non-Monotone Submodular Function

```bash
python3 -u -m python.benchmark -m obj=demo_non_monotone \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-II-b,Soma-II
```

## To see the results of the experiments

The experiment results are stored in the [`out/`](out/) folder.

## Notice

We plan on releasing this code under the MIT license, if and when accepted.
