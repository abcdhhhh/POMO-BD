from typing import Union, Callable, Optional

import torch.nn as nn

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from utils.decoding import DecodingStrategyForced
from rl4co.utils.ops import calculate_entropy
from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from .encoder import ICAMEncoder
from .decoder import ICAMDecoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class ICAMPolicy(AutoregressivePolicy):
    def __init__(
            self,
            encoder: Union[str, nn.Module] = None,
            decoder: Union[str, nn.Module] = None, 
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
        """
        You can set decode_type to multistart for symmetric problems
        """

        if pointer == "poly":
            from rl4co.models.nn.attention import PolyNetAttention
            pointer = PolyNetAttention(
                k=128,
                embed_dim=embed_dim,
                poly_layer_dim=embed_dim*2,
                num_heads=num_heads,
            )

        if encoder in ["aaft", "gt"]:
            encoder = ICAMEncoder(
                embed_dim=embed_dim,
                gt_type=encoder,
                init_embedding=init_embedding,
                env_name=env_name,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization,
                feedforward_hidden=feedforward_hidden,
                net=encoder_network,
                moe_kwargs=moe_kwargs["encoder"],
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
                moe_kwargs=moe_kwargs["encoder"],
            )
        
        if decoder == "cab":
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

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """
        Support forced actions in first few steps
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        if decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        # we pop arguments that are not part of the decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        forced_strategy = DecodingStrategyForced(decode_strategy)
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = forced_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if (actions is not None and actions.size(-1) > step) else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) duing decoding"
                )
                break

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        # Output dictionary construction
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }

        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict
