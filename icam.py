import os
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from rl4co.utils.trainer import RL4COTrainer
from rl4co.envs import get_env, ENV_REGISTRY

from models.zoo.icam.model import ICAM

@hydra.main(config_path="./configs/", config_name="icam.yaml", version_base="1.3")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.get("seed", 1234))
    L.seed_everything(cfg.get("seed", 1234))

    env_params = cfg.env
    if cfg.env_name in ENV_REGISTRY:
        env = get_env(**env_params) 
        model = ICAM(
            env=env,
            **cfg.model,
        )
    else:
        from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
        model_params_dict = OmegaConf.to_container(cfg.model, resolve=True)

        if cfg.env_name == "symop":
            from envs.symop.env import SymOPEnv
            del env_params.env_name
            env = SymOPEnv(**env_params)
            from models.nn.env_embeddings.symop import SymOPContext, SymOPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymOPInitEmbedding(cfg.embed_dim),
                "context_embedding": SymOPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })  
        elif cfg.env_name in ["sympctsp", "symspctsp"]:
            if cfg.env_name == "sympctsp":
                from envs.sympctsp.env import SymPCTSPEnv
                del env_params.env_name
                env = SymPCTSPEnv(**env_params)
            elif cfg.env_name == "symspctsp":
                from envs.symspctsp.env import SymSPCTSPEnv
                del env_params.env_name
                env = SymSPCTSPEnv(**env_params)
            from models.nn.env_embeddings.sympctsp import SymPCTSPContext, SymPCTSPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymPCTSPInitEmbedding(cfg.embed_dim),
                "context_embedding": SymPCTSPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            }) 
        elif cfg.env_name == "sympdp":
            from envs.sympdp.env import SymPDPEnv
            del env_params.env_name
            env = SymPDPEnv(**env_params)
            from models.nn.env_embeddings.sympdp import SymPDPContext, SymPDPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymPDPInitEmbedding(cfg.embed_dim),
                "context_embedding": SymPDPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            }) 
        elif cfg.env_name in ["shpp", "symshpp"]:
            if cfg.env_name == "shpp":
                from envs.shpp.env import SHPPEnv
                del env_params.env_name
                env = SHPPEnv(**env_params)
            elif cfg.env_name == "symshpp":
                from envs.symshpp.env import SymSHPPEnv
                del env_params.env_name
                env = SymSHPPEnv(**env_params)
            from models.nn.env_embeddings.shpp import SHPPContext, SHPPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SHPPInitEmbedding(cfg.embed_dim),
                "context_embedding": SHPPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })    
        model = ICAM(
            env=env,
            **model_params_dict,
        )

    if cfg.ckpt_path:
        ckpt_state_dict = torch.load(cfg.ckpt_path, weights_only=False)['state_dict']
        model.load_state_dict(ckpt_state_dict, strict=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.save_dir, "checkpoints", cfg.logger.name),
        **cfg.ckpt,
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
