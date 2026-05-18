import torch
import os
import sys
from pathlib import Path
from tensordict.tensordict import TensorDict
from rl4co.envs.routing.op.env import OPEnv 
from envs.symop.env import SymOPEnv
from models.zoo.resym.model import ReSym
from models.nn.env_embeddings.symop import SymOPContext, SymOPInitEmbedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from models.zoo.resym.transforms import CompositeAugmentation
import re

torch.manual_seed(1234)

#### READ

path = Path('./OPLib/instances/gen3')
sol_path = Path('./OPLib/solutions/ea4op/gen3')
instances = sorted([x for x in os.listdir('./OPLib/instances/gen3') if x.endswith('.oplib')], key=lambda x: int(re.search(r'(\d+)', x).group(1)))
for fn in instances:
    with open(os.path.join(path, fn), 'r') as fp:
        lines = fp.readlines()
    num_loc = 0
    max_length_origin = 0

    for i, line in enumerate(lines):
        if 'DIMENSION' in line:
            num_loc = int(line.split()[-1])
        elif 'COST_LIMIT' in line:
            max_length_origin = int(line.split()[-1])
        elif 'NODE_COORD_SECTION' in line:
            coord_i = i 
        elif 'NODE_SCORE_SECTION' in line:
            score_i = i 
        elif 'EDGE_WEIGHT_TYPE' in line:
            edge_type = line.split()[-1]
    if num_loc > 250 or edge_type != 'EUC_2D':
        continue

    locs_origin = torch.zeros(num_loc, 2, dtype=torch.float)
    prize_origin = torch.zeros(num_loc, dtype=torch.int)

    for i in range(num_loc):
        locs_origin[i][0] = float(lines[coord_i + 1 + i].split()[1])
        locs_origin[i][1] = float(lines[coord_i + 1 + i].split()[2])
        prize_origin[i] = int(lines[score_i + 1 + i].split()[1])

    scale = (locs_origin.max(dim=0)[0] - locs_origin.min(dim=0)[0]).max().item()

    locs = (locs_origin - locs_origin.min(dim=0, keepdim=True)[0]) / scale
    prize = prize_origin / 100
    max_length = max_length_origin / scale 

    with open(os.path.join(sol_path, fn.replace('.oplib', '.sol')), 'r') as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        if 'ROUTE_SCORE' in line:
            cost_bks = int(line.split()[-1])
            break


    #### INPUT

    data = TensorDict(
        {
            "locs": locs[None, 1:, :],
            "depot": locs[None, 0, :],
            "prize": prize[None, 1:],
            "max_length": torch.tensor([max_length]),
        }, 
        batch_size=1,
    )

    env = OPEnv(generator_params={"num_loc": num_loc - 1, "max_length": max_length})
    sym_env = SymOPEnv(generator_params={"num_loc": num_loc - 1, "max_length": max_length})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    td_d = env.reset(td=data.clone()).to(device)
    td_free = env.reset(td=data.clone()).to(device)
    td_bd = sym_env.reset(td=data.clone()).to(device)

    augment = CompositeAugmentation(
        num_augment_list=[8],
        augment_fn_list=['dihedral8'],
    )

    td_d = augment(td_d)
    td_free = augment(td_free)
    td_bd = augment(td_bd)

    model_d = ReSym(env=env, policy_kwargs={"encoder": "aft", "decoder": "cab", "pointer": "aft", "logit_bias": True})
    model_free = ReSym(env=env, policy_kwargs={"encoder": "aft", "decoder": "cab", "pointer": "aft", "logit_bias": True})
    model_bd = ReSym(env=sym_env, policy_kwargs={
            "encoder": "aft", "decoder": "cab", "pointer": "aft", "logit_bias": True,  
            "init_embedding": SymOPInitEmbedding(128),
            "context_embedding": SymOPContext(embed_dim=128),
            "dynamic_embedding": StaticEmbedding(),
    })

    path_d = "pretrained/op100/icam/199_30.2519_32.7912.ckpt"
    path_free = "pretrained/op100/icam-free/199_32.6061_32.9507.ckpt"
    path_bd = "pretrained/op100/icam-bd/199_31.9754_33.0976.ckpt"
    model_d.load_state_dict(torch.load(path_d, weights_only=False)['state_dict'])
    model_free.load_state_dict(torch.load(path_free, weights_only=False)['state_dict'])
    model_bd.load_state_dict(torch.load(path_bd, weights_only=False)['state_dict'])

    policy_d = model_d.policy.to(device)
    out_d = policy_d.forward(
        td_d.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_sampling", 
        num_starts=num_loc-1, 
        multistart=True,
    )
    reward = out_d['reward']
    actions = out_d['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    cost_d = int(max_reward * 100)

    policy_free = model_free.policy.to(device)
    out_free = policy_free.forward(
        td_free.clone(), 
        env, 
        phase="test", 
        decode_type="sampling", 
        num_samples=num_loc, 
        multisample=True,
    )
    reward = out_free['reward']
    actions = out_free['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    cost_free = int(max_reward * 100)


    policy_bd = model_bd.policy.to(device)
    out_bd = policy_bd.forward(
        td_bd.clone(), 
        sym_env, 
        phase="test", 
        decode_type="multistart_sampling", 
        num_starts=num_loc, 
        multistart=True,
    )
    reward = out_bd['reward']
    actions = out_bd['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    cost_bd = int(max_reward * 100)

    print(f'{fn.split('-')[0]},{cost_bks:},', end='')
    for cost in [cost_d, cost_free, cost_bd]:
        gap = -(cost - cost_bks) / cost_bks * 100
        print(f'{cost},{gap:.2f}%,', end='')
    print()
