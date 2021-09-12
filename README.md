# Stochastic Maximization of Monotone DR-Submodular Functions on the Integer Lattice - AAAI'22

## Programming Language

- Python 3.8

## Initialize the Python Environment

Install the Python virtual environment:

```bash
python3 -m venv ./venv
```

Activate the virtual environment:

```bash
source ./venv/bin/activate
```

Install third-party Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## To see the algorithms implemented

While the repository is rather extensive, the implementation of the algorithms in the [./algo](./algo) folder.
It follows the pseudocode very closely and it is commented. 

## To run the experiments:

#### Run experiments on the Synthetic DR-Monotone Submodular Function

```bash
python3 main.py -m obj=demo_monotone \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Run experiments on the Budget Allocation Problem

```bash
python3 main.py -m obj=budget_allocation \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Run experiments on the Facility Location Problem

```bash
python3 main.py -m obj=facility_location \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Run experiments on the Synthetic Non-Monotone Submodular Function

```bash
python3 main.py -m obj=demo_non_monotone \
  runtime=laptop \
  algo=SSG,SGL-I,SGL-II,SGL-II-b,Soma-II
```

## To see the results of the experiments

The experiment results are stored in the [`out/benchmarks`](out/benchmarks) folder.

## Notice

We plan on releasing this code under the MIT license, if and when accepted.
