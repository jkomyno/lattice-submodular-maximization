# Lattice Submodular Maximization

## Tech stack

- Python 3.7
- [Hydra](https://hydra.cc) configuration framework


## Initialize project

Install the virtual environment:

```bash
python -m venv ./venv
```

Activate the virtual environment:

```bash
source ./venv/bin/activate
```

Install third-party dependencies:

```bash
$ python -m pip install -r requirements.txt
```

Run:

```bash
python main.py -m obj=demo_monotone,facility_location,budget_allocation \
  algo=Soma-DR-I,Soma-II,SSG,SGL-I,SGL-II,SGL-III
```

## Configuration

Please read and edit [/conf/config.yaml](/conf/config.yaml)
