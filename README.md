# Lattice Submodular Maximization

## Programming Language

- Python 3.8

## Initialize project

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

## Run experiments

#### Synthetic DR-Monotone Submodular Function

```bash
python main.py -m obj=demo_monotone \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Budget Allocation

```bash
python main.py -m obj=budget_allocation \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Facility Location

```bash
python main.py -m obj=facility_location \
  algo=SSG,SGL-I,SGL-II,SGL-III,Soma-II,Soma-DR-I
```

#### Synthetic Non-Monotone Submodular Function

```bash
python main.py -m obj=demo_non_monotone \
  algo=SSG,SGL-I,SGL-II,SGL-II-b,Soma-II
```

## Experiment Results

The experiment results are stored in the [`out/benchmarks`](out/benchmarks) folder.

## Notice

We plan on releasing this code implementation of "" as an open-source repository, upon success of the review process.
