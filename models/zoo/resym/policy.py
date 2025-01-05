from typing import Union, Callable, Optional

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from models.zoo.icam.encoder import AFTEncoder, MatAFTEncoder, HeterAFTEncoder
from models.zoo.icam.decoder import ICAMDecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class ReSymPolicy(AutoregressivePolicy):
    def __init__(
        self,
        encoder: Union[str, nn.Module] = "aft",
        decoder: Union[str, nn.Module] = "cab", 
        embed_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        normalization: str = "instance",
        idt: bool = False,
        feedforward_hidden: int = 512,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = False,
        linear_bias_decoder: bool = False,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        pointer: Union[str, nn.Module] = None,
        temperature: float = 1.0,
        tanh_clipping: float = 50.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "sampling",
        test_decode_type: str = "sampling",
        moe_kwargs: dict = {"encoder": None, "decoder": None},
        **kwargs
    ):
        encoder_type = "mataft" if encoder in ["mataft", "matnet"] else "aft"
        if encoder == "aft":
            encoder = AFTEncoder(
                embed_dim=embed_dim,
                init_embedding=init_embedding,
                env_name=env_name,
                num_layers=num_encoder_layers,
                normalization=normalization,
                idt=idt,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
            )
        elif encoder == "mataft":
            encoder = MatAFTEncoder(
                embed_dim=embed_dim,
                env_name=env_name,
                num_layers=num_encoder_layers,
                normalization=normalization,
                idt=idt,
                feedforward_hidden=feedforward_hidden,
                init_embedding=init_embedding,
                net=encoder_network,
            )
        elif encoder == "am":
            from rl4co.models.zoo.am.encoder import AttentionModelEncoder
            encoder = AttentionModelEncoder(
                embed_dim=embed_dim,
                init_embedding=init_embedding,
                env_name=env_name,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                sdpa_fn=sdpa_fn_encoder,
                moe_kwargs=moe_kwargs["encoder"],
            )
        elif encoder == "matnet":
            from rl4co.models.zoo.matnet.encoder import MatNetEncoder
            encoder = MatNetEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                init_embedding=init_embedding,
            )
        elif encoder == "ham":
            from rl4co.models.zoo.ham.encoder import GraphHeterogeneousAttentionEncoder
            encoder = GraphHeterogeneousAttentionEncoder(
                embed_dim=embed_dim,
                init_embedding=init_embedding,
                env_name=env_name,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                sdpa_fn=sdpa_fn_encoder,
            )
        elif encoder == "haft":
            encoder = HeterAFTEncoder(
                embed_dim=embed_dim,
                init_embedding=init_embedding,
                env_name=env_name,
                num_layers=num_encoder_layers,
                normalization=normalization,
                idt=idt,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
            )
        
        if decoder == "cab":
            decoder = ICAMDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                encoder_type=encoder_type,
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
        elif decoder == "am":
            from rl4co.models.zoo.am.decoder import AttentionModelDecoder
            decoder = AttentionModelDecoder(
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
        elif decoder == "matnet":
            from rl4co.models.zoo.matnet.decoder import MatNetDecoder
            decoder = MatNetDecoder(
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

        super(ReSymPolicy, self).__init__(
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
