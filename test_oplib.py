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

torch.manual_seed(1234)

#### READ

path = Path('./OPLib/instances/gen3')
instances = sorted([x for x in os.listdir('./OPLib/instances/gen3') if x.endswith('.oplib')])
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
    print(fn)
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
    td_nd = sym_env.reset(td=data.clone()).to(device)

    augment = CompositeAugmentation(
        num_augment_list=[8],
        augment_fn_list=['dihedral8'],
    )

    td_d = augment(td_d)
    td_free = augment(td_free)
    td_nd = augment(td_nd)

    model_d = ReSym(env=env, policy_kwargs={"encoder": "aft", "decoder": "cab", "pointer": "aft", "logit_bias": True})
    model_free = ReSym(env=env, policy_kwargs={"encoder": "aft", "decoder": "cab", "pointer": "aft", "logit_bias": True})
    model_nd = ReSym(env=sym_env, policy_kwargs={
            "encoder": "aft", "decoder": "cab", "pointer": "aft", "logit_bias": True,  
            "init_embedding": SymOPInitEmbedding(128),
            "context_embedding": SymOPContext(embed_dim=128),
            "dynamic_embedding": StaticEmbedding(),
    })

    path_d = "./pretrained/op100/icam/299_30.4556_32.9341.ckpt"
    path_free = "./pretrained/op100/icam-free/299_32.7600_33.0493.ckpt"
    path_nd = "./pretrained/op100/icam-bd/299_32.1701_33.1703.ckpt"
    model_d.load_state_dict(torch.load(path_d, weights_only=False)['state_dict'])
    model_free.load_state_dict(torch.load(path_free, weights_only=False)['state_dict'])
    model_nd.load_state_dict(torch.load(path_nd, weights_only=False)['state_dict'])

    policy_d = model_d.policy.to(device)
    out_d = policy_d.forward(
        td_d.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc-1, 
        multistart=True,
    )
    reward = out_d['reward']
    actions = out_d['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    print(int(max_reward * 100))
    print(actions[max_idxs])

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
    print(int(max_reward * 100))
    print(actions[max_idxs])


    policy_nd = model_nd.policy.to(device)
    out_nd = policy_nd.forward(
        td_nd.clone(), 
        sym_env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc, 
        multistart=True,
    )
    reward = out_nd['reward']
    actions = out_nd['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    print(int(max_reward * 100))
    print(actions[max_idxs])
    # import pdb; pdb.set_trace()
