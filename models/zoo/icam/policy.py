from typing import Union, Callable

import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from .encoder import ICAMEncoder
from .decoder import ICAMDecoder

class ICAMPolicy(AutoregressivePolicy):
    def __init__(
            self,
            encoder: nn.Module = None,
            decoder: nn.Module = None, 
            embed_dim: int = 128,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            normalization: str = "instance",
            feedforward_hidden: int = 512,
            env_name: Union[str, RL4COEnvBase] = "tsp",
            encoder_network: nn.Module = None,
            init_embedding: nn.Module = None,
            context_embedding: nn.Module = None,
            dynamic_embedding: nn.Module = None,
            use_graph_context: bool = True,
            linear_bias_decoder: bool = False,
            sdpa_fn_decoder: Callable = None,
            mask_inner: bool = True,
            out_bias_pointer_attn: bool = False,
            check_nan: bool = True,
            pointer: nn.Module = None,
            temperature: float = 1.0,
            tanh_clipping: float = 50.0,
            mask_logits: bool = True,
            train_decode_type: str = "sampling",
            val_decode_type: str = "sampling",
            test_decode_type: str = "sampling",
            moe_kwargs: dict = {"encoder": None, "decoder": None},
            **kwargs
        ):
        """
        Following PolyNet, if first action is not sensitive,
        val_decode_type can be changed to "multistart_greedy" for better performance.
        """

        if encoder is None:
            encoder = ICAMEncoder(
                embed_dim=embed_dim,
                init_embedding=init_embedding,
                env_name=env_name,
                num_layers=num_encoder_layers,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                moe_kwargs=moe_kwargs["encoder"],
            )
        
        if decoder is None:
            decoder = ICAMDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                context_embedding=context_embedding,
                dynamic_embedding=dynamic_embedding,
                mask_inner=mask_inner,
                out_bias_pointer_attn=out_bias_pointer_attn,
                linear_bias=linear_bias_decoder,
                use_graph_context=use_graph_context,
                check_nan=check_nan,
                sdpa_fn=sdpa_fn_decoder,
                pointer=pointer,
                moe_kwargs=moe_kwargs["decoder"],
            )

        super(ICAMPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **kwargs
        )