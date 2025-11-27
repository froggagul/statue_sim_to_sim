# sim to sim example

## Installation
---
cloning the repository:
```
git clone https://github.com/minseokle/statue_sim_to_sim.git
cd statue_sim_to_sim
```

To create and activate the conda environment, run:
```
conda create -n sim_to_sim python=3.11 -y
conda activate sim_to_sim
```
or
```
python3.11 -m venv .venv
source .venv/bin/activate
```

Then, install the required packages:
```
pip install -r requirements.txt
```

## Usage
---
To run the simulation, execute:

```
python main.py --max_latency_step <latency step>
```