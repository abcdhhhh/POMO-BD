Please remove `load_data` from CVRPTWEnv in `rl4co/rl4co/envs/routing/cvrptw/env.py`.

# Create Env

```bash
conda create -n rl4co python=3.12
conda activate rl4co
cd rl4co
pip install -e .
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

