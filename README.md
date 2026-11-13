# Balanced Decoding of Constructive Neural Solvers for Vehicle Routing Problems

Official implementation of the paper **"Balanced Decoding of Constructive Neural Solvers for Vehicle Routing Problems"**, accepted for publication in *IEEE Transactions on Intelligent Transportation Systems* (2026), by Pengzhi Cheng, Chaoyang Wang, Jingze Li, and Weiwei Sun.

DOI: [10.1109/TITS.2026.3739787](https://doi.org/10.1109/tits.2026.3739787)

This repository is built on top of [RL4CO](https://github.com/ai4co/rl4co), which is vendored in `rl4co/` and installed in editable mode.

> ⚠️ **When using this repository's entry point (`resym.py`), the CVRPTW compatibility patch is applied automatically at runtime. If you bypass `resym.py` and import RL4CO's upstream `CVRPTWEnv` directly, follow the compatibility fix described in [Required patch: `CVRPTWEnv.load_data`](#required-patch-cvrptwenvload_data).**

## Supported problems and released configurations

| Problem | Baseline env | BD env | Configs (`configs/experiment/`) | Pretrained |
| --- | --- | --- | --- | --- |
| OP100 (maximize) | `op` (ICAM) | `symop` (ICAM-BD) | `op/icam`, `op/icam-free`, `op/icam-bd` | ✔ |
| PCTSP100 (minimize) | `pctsp` (ICAM) | `sympctsp` (ICAM-BD) | `pctsp/icam`, `pctsp/icam-free`, `pctsp/icam-bd` | ✔ |
| CVRP100 (minimize) | `cvrp` (POMO) | `symcvrp` (POMO-BD) | `cvrp/pomo`, `cvrp/pomo-free`, `cvrp/pomo-bd` | ✔ |
| CVRPTW50 (minimize) | `cvrptw` (POMO) | `symcvrptw` (POMO-BD) | `cvrptw/pomo`, `cvrptw/pomo-free`, `cvrptw/pomo-bd` | ✔ |
| TSP100 | `tsp` (AM / POMO) | – | `tsp/pomo` | – |
| MTVRP (24 variants) | `mtvrp` | `symmtvrp` | – | – |


## 1. Environment setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then create the project environment with a command matching your NVIDIA driver (check `nvidia-smi`; a separate CUDA Toolkit installation is not required for PyTorch wheels):
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then choose one command matching the CUDA version supported by your NVIDIA driver (check `nvidia-smi`; a separate CUDA Toolkit installation is not required for PyTorch wheels):

| Environment | Command |
| --- | --- |
| CUDA 11.8 | `uv sync --index https://download.pytorch.org/whl/cu118` |
| CUDA 12.4 | `uv sync --index https://download.pytorch.org/whl/cu124` |
| CUDA 13.0 | `uv sync --index https://download.pytorch.org/whl/cu130` |
| CPU only | `uv sync --index https://download.pytorch.org/whl/cpu` |

The selected index is used for PyTorch while other packages continue to use the configured package mirror. The project accepts PyTorch versions `>=2.6,<3`, so uv resolves a compatible build available from that index. To switch CUDA versions in an existing environment, rerun the corresponding command with `--upgrade-package torch`, for example: `uv sync --upgrade-package torch --index https://download.pytorch.org/whl/cu130`.

Then install optional dependencies used by logging and baseline/data utilities:

```bash
uv pip install tensorboard tsplib95 vrplib pyvrp ortools
```

`uv sync` creates `.venv`, installs the root project and the bundled RL4CO workspace member in editable mode, and uses a Python version that meets the project's `>=3.12` requirement. Run project commands with `uv run`; there is no need to activate the environment manually.

### Required patch: `CVRPTWEnv.load_data`

`rl4co/rl4co/envs/routing/cvrptw/env.py` defines a static `load_data(...)` intended for Solomon-format instances. `RL4COEnvBase.dataset()` calls `self.load_data(fpath, batch_size)`, so this override swallows the `batch_size` argument and pre-generated `.npz` datasets cannot be loaded. This repository applies the compatibility fix automatically at runtime in [`resym.py`](resym.py) by restoring the base implementation. If you bypass the project entry point and import RL4CO's upstream `CVRPTWEnv` directly, you must restore the base `load_data(...)` behavior (equivalently, remove the Solomon-specific override or keep the compatibility patch in your own startup code).

## 2. Dataset

```bash
python generate_data.py
```

This writes `.npz` files to `./data/<problem>/{val,test}/{size}.npz` (validation seed `4321`, test seed `1234`), e.g. `data/op/val/100_dist.npz`, `data/vrp/test/100.npz`, `data/vrptw/test/50.npz`, `data/pctsp/test/100.npz`, and the 24 MTVRP variants under `data/mtvrp/`.

| Problem | Sizes (validation / test) |
| --- | --- |
| `tsp` | 20 / 50 / 100 (10k), 200 (1k), 500 / 1000 / 2000 / 5000 (128), 10000 (16) |
| `vrp` (CVRP) | 20 / 50 / 100 (10k), 200 (1k), 500 (128), 1000 (100) |
| `op` | 20 / 50 / 100 (10k), 500 / 1000 (128) |
| `pctsp` | 20 / 50 / 100 (10k), 500 / 1000 (128) |
| `pdp` | 20 / 50 / 100 (10k), 500 (128) |
| `vrptw` | 20 / 50 / 100 (10k) |
| `kp` | 50 / 100 / 200 (10k) |
| `mtvrp` (24 variants) | 50 / 100 (val 128 / test 1000) |

Real-world CVRPLIB instances used for the generalization study can be downloaded from <https://galgos.inf.puc-rio.br/cvrplib/en/instances/1>. Generated data (`data/`) and experiment outputs (`outputs/`) are git-ignored.

## 3. Pretrained checkpoints

Released checkpoints are in `pretrained/`; the file name follows `{epoch}_{val/reward}_{val/max_reward}.ckpt`. Each experiment config points at its checkpoint through `ckpt_path`.

| Problem | Variant | Checkpoint |
| --- | --- | --- |
| OP100 | ICAM | `pretrained/op100/icam/199_30.2519_32.7912.ckpt` |
| OP100 | ICAM-free | `pretrained/op100/icam-free/199_32.6061_32.9507.ckpt` |
| OP100 | ICAM-BD | `pretrained/op100/icam-bd/199_31.9754_33.0976.ckpt` |
| PCTSP100 | ICAM | `pretrained/pctsp100/icam/199_-6.5306_-6.0538.ckpt` |
| PCTSP100 | ICAM-free | `pretrained/pctsp100/icam-free/199_-6.0680_-6.0244.ckpt` |
| PCTSP100 | ICAM-BD | `pretrained/pctsp100/icam-bd/199_-6.0925_-5.9797.ckpt` |
| CVRP100 | POMO | `pretrained/cvrp100/pomo/499_-16.3359_-15.9275.ckpt` |
| CVRP100 | POMO-free | `pretrained/cvrp100/pomo-free/499_-16.2794_-16.0498.ckpt` |
| CVRP100 | POMO-BD | `pretrained/cvrp100/pomo-bd/499_-16.3031_-15.9040.ckpt` |
| CVRPTW50 | POMO | `pretrained/cvrptw50/pomo/499_-16.1163_-15.3068.ckpt` |
| CVRPTW50 | POMO-free | `pretrained/cvrptw50/pomo-free/499_-15.8970_-15.5910.ckpt` |
| CVRPTW50 | POMO-BD | `pretrained/cvrptw50/pomo-bd/499_-15.9947_-15.2371.ckpt` |

## 4. Evaluation (quick start)

Take ICAM-BD on OP as an example. Hyper-parameters live in `configs/experiment/op/icam-bd.yaml` (note: `experiment`, singular), which inherits `configs/experiment/op/icam.yaml`:

```bash
# test with the released checkpoint
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=op/icam-bd
```

Other configurations are selected the same way:

```bash
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=pctsp/icam-bd
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=cvrp/pomo-bd
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=cvrptw/pomo-bd
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=cvrp/pomo-free   # free first step
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=cvrp/pomo        # multi-start baseline
```

`mode: test` is the default; `ckpt_path` is taken from the config. To test your own checkpoint, override it:

```bash
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=op/icam-bd \
    ckpt_path=outputs/checkpoints/op100/bd/<logger_name>/<epoch>_<reward>_<max_reward>.ckpt
```

The runner prints the metrics, the wall-clock time of the test pass, and the peak GPU memory.

## 5. Training

```bash
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=op/icam-bd \
    mode=train trainer.logger=tensorboard ckpt_path=null
```

- `ckpt_path=null` starts from scratch; pass a checkpoint path to resume.
- Checkpoints are written to `outputs/checkpoints/<logger_name>/` (`ckpt.monitor=val/max_reward`, saved every 50 epochs + `last`).
- Logs are written under `outputs/<problem><num_loc>/...` by `TensorBoardLogger`:

```bash
tensorboard --logdir outputs            # everything
tensorboard --logdir outputs/op100      # only OP100
```

## 6. Configuration

Experiments are configured with Hydra. Every file under `configs/experiment/` is declared as `# @package _global_`, so a variant config simply inherits a base config and overrides a few keys:

- `tsp/pomo.yaml` — base config for the AM/POMO backbone (encoder `am`, decoder `am`, `pointer: null`); `cvrp/pomo.yaml` and `cvrptw/pomo.yaml` inherit from it.
- `op/icam.yaml` — base config for the ICAM backbone (encoder `aft`, decoder `cab`, `pointer: aft`); `pctsp/icam.yaml` inherits from it.
- `<model>-free.yaml` / `<model>-bd.yaml` — switch `env_name` to the symmetric (BD) environment, set `num_starts`, the decoding types and the logger name.

Frequently used options (all overridable on the command line):

| Key | Meaning |
| --- | --- |
| `mode` | `train` / `val` / `test` |
| `env_name` | `op`, `pctsp`, `cvrp`, `cvrptw`, `tsp` for the baselines; `symop`, `sympctsp`, `symcvrp`, `symcvrptw` for Balanced Decoding |
| `num_loc` | instance size (the corresponding `.npz` files must exist) |
| `env.val_file` / `env.test_file` | dataset split(s) used for validation / testing |
| `model.num_starts` | number of sampled tours: `null` lets the environment decide (`N` for the baseline envs, `N+1` for the BD envs); an explicit value such as `101` / `51` reproduces the `*-free` variants |
| `model.policy_kwargs.train_decode_type` / `val_decode_type` | `multistart_sampling` for BD (symmetric) problems, `sampling` for the free-first-step variants |
| `model.policy_kwargs.encoder` / `decoder` / `pointer` | backbone components: `aft` + `cab` + `aft` (ICAM), `am` + `am` + `null` (POMO/AM); other options are `mataft`, `am`, `matnet` (encoder) and `matnet`, `vam` (decoder) |
| `model.num_augment_list` / `model.augment_fn_list` | test-time instance augmentation, e.g. `[8]` with `['dihedral8']` |
| `model.batch_size` / `model.val_batch_size` | `128` / `500` (kept identical across variants for a fair comparison) |
| `trainer.max_epochs`, `trainer.precision`, `trainer.logger` | training schedule, `16-mixed` by default, `null`/`tensorboard`/`wandb` |
| `save_dir` | root of logs and checkpoints (default `./outputs/`) |

Example: evaluate ICAM-BD on OP with a different size and logging backend:

```bash
CUDA_VISIBLE_DEVICES=0 python resym.py +experiment=op/icam-bd \
    num_loc=50 env.val_file=[op/val/50_dist.npz] env.test_file=[op/test/50_dist.npz] \
    trainer.logger=tensorboard
```

## 7. Repository structure

```text
POMO-BD/
├── resym.py                  # Hydra entry point (train / val / test)
├── generate_data.py          # dataset generation (.npz)
├── configs/
│   ├── resym.yaml            # Hydra root config
│   └── experiment/           # one yaml per experiment (Hydra group `experiment`)
│       ├── tsp/pomo.yaml     #   base: AM encoder + AM decoder (POMO / AM)
│       ├── cvrp/{pomo,pomo-free,pomo-bd}.yaml
│       ├── cvrptw/{pomo,pomo-free,pomo-bd}.yaml
│       ├── op/{icam,icam-free,icam-bd}.yaml      #   base: AFT encoder + CAB decoder (ICAM)
│       └── pctsp/{icam,icam-free,icam-bd}.yaml
├── envs/                     # symmetric (Balanced Decoding) environments
│   ├── symop/ sympctsp/ symcvrp/ symcvrptw/
│   └── mtvrp/ symmtvrp/ ...
├── models/
│   ├── zoo/resym/            # ReSym: model, policy, symmetricity/mixed losses, augmentations
│   ├── zoo/icam/             # AFT encoder and ICAM (CAB) decoder
│   └── nn/env_embeddings/     # init / context embeddings for every environment
├── pretrained/               # released checkpoints
└── rl4co/                    # vendored RL4CO, installed editable by `uv sync`
```

## Citation

If you use this code, please cite:

```bibtex
@article{cheng2026balanced,
  title   = {Balanced Decoding of Constructive Neural Solvers for Vehicle Routing Problems},
  author  = {Cheng, Pengzhi and Wang, Chaoyang and Li, Jingze and Sun, Weiwei},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  doi     = {10.1109/TITS.2026.3739787},
  url     = {https://doi.org/10.1109/tits.2026.3739787}
}
```

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)). The vendored `rl4co/` copy remains under its original MIT license by the RL4CO authors.

## Acknowledgements

This implementation builds on [RL4CO](https://github.com/ai4co/rl4co) (MIT License). We thank its authors and the community for maintaining such a useful benchmark framework.
