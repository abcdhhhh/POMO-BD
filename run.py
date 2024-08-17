import os
import hydra
from omegaconf import OmegaConf, DictConfig

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

    if model_params_dict['policy_kwargs']['pointer'] == "poly":
        from rl4co.models.nn.attention import PolyNetAttention
        model_params_dict['policy_kwargs']['pointer'] = PolyNetAttention(
            **cfg.poly, 
            poly_layer_dim=cfg.embed_dim*2
        )

    if model_params_dict['policy_kwargs']['encoder'] == "am":
        from rl4co.models.zoo.am.encoder import AttentionModelEncoder
        model_params_dict['policy_kwargs']['encoder'] = AttentionModelEncoder(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=model_params_dict['policy_kwargs']['num_encoder_layers'],
            normalization=model_params_dict['policy_kwargs']['normalization'],
        )

    model = ICAM(
        env=env,
        **model_params_dict,
    )

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
        trainer.fit(model, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "val":
        trainer.validate(model, ckpt_path=cfg.ckpt_path)

if __name__ == "__main__":
    run()
