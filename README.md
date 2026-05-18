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

CVRPlib instances can be downloaded from https://galgos.inf.puc-rio.br/cvrplib/en/instances/1 .

# Run

Take ICAM-BD on OP as an example, set hyperparameters in `configs/experiments/op/icam-bd.yaml`, and then run

```bash
# test
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=op/icam-bd

# train
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=op/icam-bd mode=train trainer.logger=tensorboard ckpt_path=null
```

# View

```bash
tensorboard --logdir outputs/op100
```
