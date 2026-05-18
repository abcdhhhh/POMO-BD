import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.routing.cvrp.env import CVRPEnv 
from envs.symcvrp.env import SymCVRPEnv
from models.zoo.resym.model import ReSym
from models.nn.env_embeddings.symcvrp import SymVRPContext, SymVRPInitEmbedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from models.zoo.resym.transforms import CompositeAugmentation
import vrplib
import os 
from tqdm import tqdm

torch.manual_seed(1234)

def get_reward_unnormalized(td, actions, sym=False):
    from rl4co.utils.ops import gather_by_index
    locs = td["locs"]
    if sym:
        ordered_locs = gather_by_index(locs, actions).float()
    else:
        ordered_locs = torch.cat(
            [
                locs[..., 0:1, :],  # depot
                gather_by_index(locs, actions),  # order locations
            ],
            dim=1,
        ).float()
    ordered_locs_next = torch.roll(ordered_locs, -1, dims=-2)
    d = (ordered_locs - ordered_locs_next).norm(dim=-1, p=2)
    # if locs dtype is int, the round, else don't round
    if locs.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        d = d.round()
    return -d.sum(dim=-1).round().int().item()

# Utils function: we will normalize the coordinates of the VRP instances
def normalize_coord(coord:torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    scaler = max(x_max-x_min, y_max-y_min)
    x_scaled = (x - x_min) / scaler
    y_scaled = (y - y_min) / scaler
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled

def vrplib_to_td(problem, normalize=True):
    coords = torch.tensor(problem['node_coord'])
    coords_norm = normalize_coord(coords) if normalize else coords
    demand = torch.tensor(problem['demand'][1:]).float()
    capacity = problem['capacity']
    n = coords.shape[0]
    td = TensorDict({
        'depot': coords_norm[0,:],
        'locs': coords_norm[1:,:],
        'demand': demand / capacity, # normalized demand
        'capacity': capacity, # original capacity, not needed for inference
    })
    td = td[None] # add batch dimension, in this case just 1
    return td

# problem_names = vrplib.list_names(low=50, high=250, vrp_type='cvrp') 

# instances = [] 
# for name in problem_names:
#     if 'X' in name:
#         instances.append(name)
    # if 'A' in name:
    #     instances.append(name)
    # elif 'B' in name:
    #     instances.append(name)
    # elif 'E' in name:
    #     instances.append(name)
    # elif 'F' in name:
    #     instances.append(name)
    # elif 'M' in name and 'CMT' not in name:
    #     instances.append(name)

instances = sorted([x[:-4] for x in os.listdir('./vrplib/X/') if x.endswith('.vrp')])

# Modify the path you want to save 
# Note: we don't have to create this folder in advance
path_to_save = './vrplib/X/' 

# os.makedirs(path_to_save, exist_ok=True)
# for instance in tqdm(instances):
#     vrplib.download_instance(instance, path_to_save)
#     vrplib.download_solution(instance, path_to_save)

for instance in instances:
    # Inference
    problem = vrplib.read_instance(os.path.join(path_to_save, instance+'.vrp'))
    td = vrplib_to_td(problem)
    td_orginal = vrplib_to_td(problem, normalize=False)
    solution = vrplib.read_solution(os.path.join(path_to_save, instance+'.sol'))
    optimal_cost = solution['cost']


    num_loc = td['locs'].shape[1] + 1
    capacity = problem['capacity']
    if num_loc - 1 > 250:
        continue

    env = CVRPEnv(generator_params={"num_loc": num_loc - 1, "capacity": capacity})
    sym_env = SymCVRPEnv(generator_params={"num_loc": num_loc - 1, "capacity": capacity})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    td_forced = env.reset(td=td.clone()).to(device)
    td_free = env.reset(td=td.clone()).to(device)
    td_bd = sym_env.reset(td=td.clone()).to(device)

    augment = CompositeAugmentation(
        num_augment_list=[8],
        augment_fn_list=['dihedral8'],
    )

    td_forced = augment(td_forced)
    td_free = augment(td_free)
    td_bd = augment(td_bd)

    model_forced = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None})
    model_free = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None})
    model_bd = ReSym(env=sym_env, policy_kwargs={
            "encoder": "am", "decoder": "am", "pointer": None,     
            "init_embedding": SymVRPInitEmbedding(128),
            "context_embedding": SymVRPContext(embed_dim=128),
            "dynamic_embedding": StaticEmbedding(),
    })

    path_forced = "./pretrained/cvrp100/pomo/499_-16.3359_-15.9275.ckpt"
    path_free = "./pretrained/cvrp100/pomo-free/499_-16.2794_-16.0498.ckpt"
    path_bd = "./pretrained/cvrp100/pomo-bd/499_-16.3031_-15.9040.ckpt"
    model_forced.load_state_dict(torch.load(path_forced, weights_only=False)['state_dict'])
    model_free.load_state_dict(torch.load(path_free, weights_only=False)['state_dict'])
    model_bd.load_state_dict(torch.load(path_bd, weights_only=False)['state_dict'])

    policy_forced = model_forced.policy.to(device)
    out_forced = policy_forced.forward(
        td_forced.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_sampling", 
        num_starts=num_loc-1, 
        multistart=True,
    )
    reward = out_forced['reward']
    actions = out_forced['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone()).to(device)
    cost_forced = -get_reward_unnormalized(unnormalized_td, action)
    

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
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone()).to(device)
    cost_free = -get_reward_unnormalized(unnormalized_td, action)
    


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
    action = actions[max_idxs][None, :]
    unnormalized_td = sym_env.reset(td_orginal.clone()).to(device)
    cost_bd = -get_reward_unnormalized(unnormalized_td, action, sym=True)
    
    print(f'{instance},{optimal_cost:},', end='')
    for cost in [cost_forced, cost_free, cost_bd]:
        gap = (cost - optimal_cost) / optimal_cost * 100
        print(f'{cost:},{gap:.2f}%,', end='')
    print()
    
 