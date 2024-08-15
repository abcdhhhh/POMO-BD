from rl4co.envs import get_env

from rl4co.models.zoo import POMO, SymNCO, NARGNNPolicy, AttentionModel
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.nn.attention import PolyNetAttention
from rl4co.utils.trainer import RL4COTrainer
from models.zoo.icam.model import ICAM
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import torch
import lightning as L

L.seed_everything(1234)

num_loc = 100
elite_k = 5
elite_alpha = 1.
save_dir = "./lightning_logs/"
env_name = "tsp"
name = f"icam_{env_name}{num_loc}_poly_{elite_k}_{elite_alpha}"
env = get_env(env_name=env_name, generator_params=dict(num_loc=num_loc), check_solution=False)

pointer = PolyNetAttention(
    k=num_loc, 
    embed_dim=128, 
    poly_layer_dim=256,
    num_heads=8,
)

model = ICAM(
    env, 
    train_data_size=160_000,
    val_data_size=10_000,
    batch_size=64,
    val_batch_size=500,
    num_augment=8,
    metrics={
        "train": ["loss", "reward", "max_reward"],
        "val": ["reward", "max_reward", "max_aug_reward"],
        "test": ["reward", "max_reward", "max_aug_reward"],
    }, 
    # policy=NARGNNPolicy(),
    policy_kwargs={
        "val_decode_type": "greedy",
        "test_decode_type": "greedy",
        "pointer": pointer,
        "tanh_clipping": 50.,
    },
    optimizer_kwargs={
        "lr": 1e-4,
        "weight_decay": 1e-6,
    },
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs={
        "milestones": [80, 95],
        "gamma": 0.1,
    },
    elite_k=elite_k,
    elite_alpha=elite_alpha,
)

logger = WandbLogger(project="rl4co", name=name, save_dir=save_dir)

checkpoint_callback = ModelCheckpoint(
    monitor="val/reward",
    dirpath=os.path.join(save_dir, "checkpoints", name),
    filename="{epoch:02d}_{val/reward:.4f}",
    auto_insert_metric_name=False,
    save_top_k=3,
    save_last=True,
    mode="max",
)
 
trainer = RL4COTrainer(
    max_epochs=100,
    logger=logger,
    precision="32-true",
    callbacks=[checkpoint_callback],
)


trainer.fit(
    model, 
)
