import os
import hydra
from omegaconf import OmegaConf, DictConfig

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from rl4co.utils.trainer import RL4COTrainer
from rl4co.envs import get_env, ENV_REGISTRY

from models.zoo.resym.model import ReSym

@hydra.main(config_path="configs", config_name="resym", version_base="1.3")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.get("seed", 1234))
    L.seed_everything(cfg.get("seed", 1234))

    env_params = cfg.env
    if cfg.env_name in ENV_REGISTRY:
        from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
        model_params_dict = OmegaConf.to_container(cfg.model, resolve=True)
        if cfg.env_name == "cvrptw":
            from envs.symcvrptw.generator import SymCVRPTWGenerator
            generator = SymCVRPTWGenerator(**env_params.generator_params)
            env = get_env(generator=generator, **env_params)
        elif cfg.env_name == "op":
            from envs.symop.generator import SymOPGenerator
            generator = SymOPGenerator(**env_params.generator_params)
            env = get_env(generator=generator, **env_params)
        elif cfg.env_name == "pctsp":
            from envs.sympctsp.generator import SymPCTSPGenerator
            generator = SymPCTSPGenerator(**env_params.generator_params)
            env = get_env(generator=generator, **env_params)
        elif cfg.env_name == "mtvrp":
            from envs.mtvrp.env import MTVRPEnv
            del env_params.env_name
            env = MTVRPEnv(**env_params)
            from models.nn.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder, MTVRPContextEmbeddingRouteFinder
            model_params_dict["policy_kwargs"].update({
                "init_embedding": MTVRPInitEmbeddingRouteFinder(cfg.embed_dim),
                "context_embedding": MTVRPContextEmbeddingRouteFinder(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })
        else:
            env = get_env(**env_params) 
        model = ReSym(
            env=env,
            **model_params_dict,
        )
    else:
        from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
        model_params_dict = OmegaConf.to_container(cfg.model, resolve=True)

        if cfg.env_name == "kp":
            from envs.kp.env import KPEnv
            del env_params.env_name
            env = KPEnv(**env_params)
            from models.nn.env_embeddings.kp import KPContext, KPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": KPInitEmbedding(cfg.embed_dim),
                "context_embedding": KPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })
        elif cfg.env_name == "symmtvrp":
            from envs.symmtvrp.env import SymMTVRPEnv
            del env_params.env_name
            env = SymMTVRPEnv(**env_params)
            from models.nn.env_embeddings.symmtvrp import SymMTVRPInitEmbeddingRouteFinder, SymMTVRPContextEmbeddingRouteFinder
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymMTVRPInitEmbeddingRouteFinder(cfg.embed_dim),
                "context_embedding": SymMTVRPContextEmbeddingRouteFinder(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })
        elif cfg.env_name == "symop":
            from envs.symop.env import SymOPEnv
            del env_params.env_name
            env = SymOPEnv(**env_params)
            from models.nn.env_embeddings.symop import SymOPContext, SymOPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymOPInitEmbedding(cfg.embed_dim),
                "context_embedding": SymOPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })  
        elif cfg.env_name == "sympctsp":
            from envs.sympctsp.env import SymPCTSPEnv
            del env_params.env_name
            env = SymPCTSPEnv(**env_params)
            from models.nn.env_embeddings.sympctsp import SymPCTSPContext, SymPCTSPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymPCTSPInitEmbedding(cfg.embed_dim),
                "context_embedding": SymPCTSPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            }) 
        elif cfg.env_name == "symcvrp":
            from envs.symcvrp.env import SymCVRPEnv
            del env_params.env_name
            env = SymCVRPEnv(**env_params)
            from models.nn.env_embeddings.symcvrp import SymVRPContext, SymVRPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymVRPInitEmbedding(cfg.embed_dim),
                "context_embedding": SymVRPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            })
        elif cfg.env_name == "symcvrptw":
            from envs.symcvrptw.env import SymCVRPTWEnv
            del env_params.env_name
            env = SymCVRPTWEnv(**env_params)
            from models.nn.env_embeddings.symcvrptw import SymVRPTWContext, SymVRPTWInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": SymVRPTWInitEmbedding(cfg.embed_dim),
                "context_embedding": SymVRPTWContext(embed_dim=cfg.embed_dim),
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
        elif cfg.env_name == "stsp":
            from envs.stsp.env import STSPEnv
            del env_params.env_name
            env = STSPEnv(**env_params)
            from models.nn.env_embeddings.stsp import STSPContext, STSPInitEmbedding
            model_params_dict["policy_kwargs"].update({
                "init_embedding": STSPInitEmbedding(cfg.embed_dim),
                "context_embedding": STSPContext(embed_dim=cfg.embed_dim),
                "dynamic_embedding": StaticEmbedding(),
            }) 
        model = ReSym(
            env=env,
            **model_params_dict,
        )

    # if cfg.ckpt_path:
    #     ckpt_state_dict = torch.load(cfg.ckpt_path, weights_only=False)['state_dict']
    #     model.load_state_dict(ckpt_state_dict, strict=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.save_dir, "checkpoints", cfg.logger.name),
        **cfg.ckpt,
    )

    trainer_params_dict = OmegaConf.to_container(cfg.trainer, resolve=True)
    if trainer_params_dict['logger'] == 'wandb':
        trainer_params_dict['logger'] = WandbLogger(**cfg.logger)
    elif trainer_params_dict['logger'] == 'tensorboard':
        trainer_params_dict['logger'] = TensorBoardLogger(**cfg.logger)

    trainer = RL4COTrainer(
        **trainer_params_dict,
        callbacks=[checkpoint_callback],
    )
    
    if cfg.mode == "train":
        trainer.fit(model, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "val":
        trainer.validate(model, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "test":
        trainer.test(model, ckpt_path=cfg.ckpt_path)

if __name__ == "__main__":
    run()
