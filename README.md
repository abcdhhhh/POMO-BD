Please remove `load_data` from CVRPTWEnv in `rl4co/rl4co/envs/routing/cvrptw/env.py`.

# Create Env

```bash
conda create -n rl4co python=3.12
conda activate rl4co
cd rl4co
pip install -e .
pip install tensorboard tsplib95 vrplib pyvrp ortools
```

if cuda version is 11, then reinstall pytorch by

```bash
pip install torch -i https://download.pytorch.org/whl/cu118 --force-reinstall
```

# Generate Data

```bash
python generate_data.py
```

# Run

modify `configs/resym.yaml`, and then run

```bash
python resym.py
```

# View

```bash
tensorboard --logdir outputs/cvrp50 --port 6006
```
