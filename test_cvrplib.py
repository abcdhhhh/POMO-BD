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

def get_reward_unnormalized(td, actions):
    from rl4co.utils.ops import gather_by_index
    locs = td["locs"]
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

instances = sorted([x[:-4] for x in os.listdir('./vrplib/') if x.endswith('.vrp')])

# Modify the path you want to save 
# Note: we don't have to create this folder in advance
path_to_save = './vrplib/' 

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

    env = CVRPEnv(generator_params={"num_loc": num_loc - 1, "capacity": capacity})
    sym_env = SymCVRPEnv(generator_params={"num_loc": num_loc - 1, "capacity": capacity})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # td_d = env.reset(td=td.clone()).to(device)
    # td_free = env.reset(td=td.clone()).to(device)
    # td_nd = sym_env.reset(td=td.clone()).to(device)

    # augment = CompositeAugmentation(
    #     num_augment_list=[8],
    #     augment_fn_list=['dihedral8'],
    # )

    # td_d = augment(td_d)
    # td_free = augment(td_free)
    # td_nd = augment(td_nd)

    # model_d = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None})
    # model_free = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None})
    # model_nd = ReSym(env=sym_env, policy_kwargs={
    #         "encoder": "am", "decoder": "am", "pointer": None,     
    #         "init_embedding": SymVRPInitEmbedding(128),
    #         "context_embedding": SymVRPContext(embed_dim=128),
    #         "dynamic_embedding": StaticEmbedding(),
    # })

    # path_d = "./pretrained/pomo-forced-cvrp100/499_-16.3359_-15.9275.ckpt"
    # path_free = "./pretrained/pomo-free-cvrp100/499_-16.2794_-16.0498.ckpt"
    # path_nd = "./pretrained/pomo-symcvrp100/499_-16.3031_-15.9040.ckpt"
    # model_d.load_state_dict(torch.load(path_d, weights_only=False)['state_dict'])
    # model_free.load_state_dict(torch.load(path_free, weights_only=False)['state_dict'])
    # model_nd.load_state_dict(torch.load(path_nd, weights_only=False)['state_dict'])

    # policy_d = model_d.policy.to(device)
    # out_d = policy_d.forward(
    #     td_d.clone(), 
    #     env, 
    #     phase="test", 
    #     decode_type="multistart_sampling", 
    #     num_starts=num_loc-1, 
    #     multistart=True,
    # )
    # reward = out_d['reward']
    # actions = out_d['actions']
    # max_reward, max_idxs = reward.max(dim=-1)
    # action = actions[max_idxs][None, :]
    # unnormalized_td = env.reset(td_orginal.clone())
    # cost_d = -env.get_reward(unnormalized_td, action).int().item()
    

    # policy_free = model_free.policy.to(device)
    # out_free = policy_free.forward(
    #     td_free.clone(), 
    #     env, 
    #     phase="test", 
    #     decode_type="sampling", 
    #     num_samples=num_loc, 
    #     multisample=True,
    # )
    # reward = out_free['reward']
    # actions = out_free['actions']
    # max_reward, max_idxs = reward.max(dim=-1)
    # action = actions[max_idxs][None, :]
    # unnormalized_td = env.reset(td_orginal.clone())
    # cost_free = -env.get_reward(unnormalized_td, action).int().item()
    


    # policy_nd = model_nd.policy.to(device)
    # out_nd = policy_nd.forward(
    #     td_nd.clone(), 
    #     sym_env, 
    #     phase="test", 
    #     decode_type="multistart_sampling", 
    #     num_starts=num_loc, 
    #     multistart=True,
    # )
    # reward = out_nd['reward']
    # actions = out_nd['actions']
    # max_reward, max_idxs = reward.max(dim=-1)
    # action = actions[max_idxs][None, :]
    # unnormalized_td = sym_env.reset(td_orginal.clone())
    # cost_nd = -sym_env.get_reward(unnormalized_td, action).int().item()
    
    # print(f'{instance},{optimal_cost:},', end='')
    # for cost in [cost_d, cost_free, cost_nd]:
    #     gap = (cost - optimal_cost) / optimal_cost * 100
    #     print(f'{cost:},{gap:.2f}%,', end='')
    # print()
    
    td_am = env.reset(td=td.clone()).to(device)
    td_vam = env.reset(td=td.clone()).to(device)
    td_vam_false = env.reset(td=td.clone()).to(device)

    augment = CompositeAugmentation(
        num_augment_list=[8],
        augment_fn_list=['dihedral8'],
    )

    td_am = augment(td_am)
    td_vam = augment(td_vam)
    td_vam_false = augment(td_vam_false)

    model_am = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None, "context_k": 0, "tanh_clipping": 50.0, })
    model_vam = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "vam", "pointer": None, "context_k": 0, "tanh_clipping": 50.0, "logit_bias": True,})
    model_vam_false = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "vam", "pointer": None, "context_k": 0, "tanh_clipping": 50.0, "logit_bias": False, })
    
    path_am = "outputs/checkpoints/cvrp100/am-am-None-100[]-multistart_sampling-0-False-0.0-v1/last.ckpt"
    path_vam = "outputs/checkpoints/cvrp100/am-vam-None-100[]-multistart_sampling-0-True-0.0-v1/last.ckpt"
    path_vam_false = "outputs/checkpoints/cvrp100/am-vam-None-100[]-multistart_sampling-0-False-0.0-v1/last.ckpt"
    model_am.load_state_dict(torch.load(path_am, weights_only=False)['state_dict'])
    model_vam.load_state_dict(torch.load(path_vam, weights_only=False)['state_dict'])
    model_vam_false.load_state_dict(torch.load(path_vam_false, weights_only=False)['state_dict'])

    policy_am = model_am.policy.to(device)
    out_am = policy_am.forward(
        td_am.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc-1, 
        multistart=True,
    )
    reward = out_am['reward']
    actions = out_am['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone()).to(device)
    cost_am = -get_reward_unnormalized(unnormalized_td, action)

    policy_vam = model_vam.policy.to(device)
    out_vam = policy_vam.forward(
        td_vam.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc-1, 
        multistart=True,
    )
    reward = out_vam['reward']
    actions = out_vam['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone()).to(device)
    cost_vam = -get_reward_unnormalized(unnormalized_td, action)

    policy_vam_false = model_vam_false.policy.to(device)
    out_vam_false = policy_vam_false.forward(
        td_vam_false.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc-1, 
        multistart=True,
    )
    reward = out_vam_false['reward']
    actions = out_vam_false['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone()).to(device)
    cost_vam_false = -get_reward_unnormalized(unnormalized_td, action)

    print(f'{instance},{optimal_cost:},{cost_am:},{(cost_am - optimal_cost) / optimal_cost * 100:.2f}%,{cost_vam:},{(cost_vam - optimal_cost) / optimal_cost * 100:.2f}%,{cost_vam_false:},{(cost_vam_false - optimal_cost) / optimal_cost * 100:.2f}%,')