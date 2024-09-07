import os
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from rl4co.utils.trainer import RL4COTrainer
from rl4co.envs import get_env

from models.zoo.icam.model import ICAM

@hydra.main(config_path="./configs/", config_name="icam.yaml", version_base="1.3")
def run(cfg: DictConfig):
    L.seed_everything(cfg.get("seed", 1234))

    env_params = cfg.env
    model_params_dict = OmegaConf.to_container(cfg.model, resolve=True)
    if env_params.env_name == "shpp":
        from envs.routing.shpp.env import SHPPEnv
        from models.nn.env_embeddings.init import SHPPInitEmbedding
        from models.nn.env_embeddings.context import SHPPContext
        from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
        del env_params.env_name
        env = SHPPEnv(**env_params)
        
        model_params_dict['policy_kwargs'].update({
            "init_embedding": SHPPInitEmbedding(embed_dim=cfg.embed_dim),
            "context_embedding": SHPPContext(embed_dim=cfg.embed_dim),
            "dynamic_embedding": StaticEmbedding(),
        })
    else:
        env = get_env(**env_params)

    model = ICAM(
        env=env,
        **model_params_dict,
    )
    if cfg.ckpt_path:
        ckpt_state_dict = torch.load(cfg.ckpt_path)['state_dict']
        model.load_state_dict(ckpt_state_dict)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/max_reward",
        dirpath=os.path.join(cfg.save_dir, "checkpoints", cfg.logger.name),
        filename="{epoch:02d}_{val/reward:.4f}_{val/max_reward:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    trainer_params_dict = OmegaConf.to_container(cfg.trainer, resolve=True)
    if trainer_params_dict['logger'] == 'wandb':
        trainer_params_dict['logger'] = WandbLogger(**cfg.logger)
    
    trainer = RL4COTrainer(
        **trainer_params_dict,
        callbacks=[checkpoint_callback],
    )
    
    if cfg.mode == "train":
        trainer.fit(model)
    elif cfg.mode == "val":
        trainer.validate(model)
    elif cfg.mode == "test":
        trainer.test(model)

if __name__ == "__main__":
    run()
