import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.routing.tsp.env import TSPEnv
from models.zoo.resym.model import ReSym
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from models.zoo.resym.transforms import CompositeAugmentation
import requests, tarfile, os, gzip, shutil
from tqdm.auto import tqdm
from tsplib95.loaders import load_problem, load_solution
import os 
from tqdm import tqdm
import re

torch.manual_seed(1234)

# def download_and_extract_tsplib(url, directory="tsplib", delete_after_unzip=True):
#     os.makedirs(directory, exist_ok=True)
    
#     # Download with progress bar
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         total_size = int(r.headers.get('content-length', 0))
#         with open("tsplib.tar.gz", 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
#             for chunk in r.iter_content(8192):
#                 f.write(chunk)
#                 pbar.update(len(chunk))

#     # Extract tar.gz
#     with tarfile.open("tsplib.tar.gz", 'r:gz') as tar:
#         tar.extractall(directory)

#     # Decompress .gz files inside directory
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".gz"):
#                 path = os.path.join(root, file)
#                 with gzip.open(path, 'rb') as f_in, open(path[:-3], 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
#                 os.remove(path)

#     if delete_after_unzip:
#         os.remove("tsplib.tar.gz")

# # Download and extract all tsp files under tsplib directory
# download_and_extract_tsplib("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz")

def get_reward_unnormalized(td, actions):
    from rl4co.utils.ops import gather_by_index
    locs = td['locs']
    ordered_locs = gather_by_index(locs, actions).float()
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
    scaler = max(x_max - x_min, y_max - y_min)
    x_scaled = (x - x_min) / scaler
    y_scaled = (y - y_min) / scaler
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled 

def tsplib_to_td(problem, normalize=True):
    coords = torch.tensor(problem['node_coords'])
    coords_norm = normalize_coord(coords) if normalize else coords
    td = TensorDict({
        'locs': coords_norm,
    })
    td = td[None] # add batch dimension, in this case just 1
    return td

# Load the problem from TSPLib
tsplib_dir = './TSPLIB/res'# modify this to the directory of your prepared files
files = os.listdir(tsplib_dir)
problem_files_full = sorted([file for file in files if file.endswith('.tsp')])

# Load the optimal solution files from TSPLib
solution_files = sorted([file for file in files if file.endswith('.opt.tour')])

problems = []

for prob_file in problem_files_full:
    sol_file = prob_file.replace('.tsp', '.opt.tour')
    problem = load_problem(os.path.join(tsplib_dir, prob_file))

    # NOTE: in some problem files (e.g. hk48), the node coordinates are not available
    # we temporarily skip these problems
    if not len(problem.node_coords) or problem.edge_weight_type != 'EUC_2D':
        continue
    
    node_coords = torch.tensor([v for v in problem.node_coords.values()])
    try:
        solution = load_solution(os.path.join(tsplib_dir, sol_file)).tours[0]
    except FileNotFoundError:
        solution = None

    problems.append({
        "name": sol_file.replace('.opt.tour', ''),
        "node_coords": node_coords,
        "solution": solution,
        "dimension": problem.dimension
    })
    
    
# order by dimension
problems = sorted(problems, key=lambda x: x['dimension'])

with open('TSPLIB/res/_Optimal solutions for symmetric TSPs.value', 'r', encoding='utf-8') as f:
    lines = f.readlines()

optimal_dict = {}

pattern = re.compile(r'^\s*(\w+)\s*:\s*(\d+)(?:\s*\((.*?)\))?')

for line in lines:
    match = pattern.match(line.strip())
    if match:
        name = match.group(1)     # 问题名，如 a280
        value = int(match.group(2)) # 最优值，如 2579
        note = match.group(3)      # 备注，如 EUC_2D
        
        # 如果有备注（如 dsj1000），则把备注加到 key 里，方便区分
        if note:
            full_key = f"{name}({note})"
            optimal_dict[full_key] = value
            # 同时保留一个不带括号的通用 key（如果有多个，后者会覆盖前者）
            if name not in optimal_dict:
                optimal_dict[name] = value
        else:
                optimal_dict[name] = value
                

for problem in problems:
    instance = problem['name']
    # Inference
    td = tsplib_to_td(problem)
    td_orginal = tsplib_to_td(problem, normalize=False)

    num_loc = td['locs'].shape[1]

    if num_loc <= 0 or num_loc > 250:
        continue

    env = TSPEnv(generator_params={"num_loc": num_loc})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        bks_sol = (torch.tensor(problem['solution'], device=device, dtype=torch.int64) - 1)[None]
        unnormalized_td = env.reset(td_orginal.clone()).to(device)
        optimal_cost = -get_reward_unnormalized(unnormalized_td, bks_sol)
        assert optimal_cost == optimal_dict[instance], f"Optimal cost mismatch for {instance}: {optimal_cost} vs {optimal_dict[instance]}"
    except:
        optimal_cost = optimal_dict.get(instance, None)
    
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

    model_am = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None, "tanh_clipping": 50.0, })
    model_vam = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "vam", "pointer": None, "tanh_clipping": 50.0, "logit_bias": True, })
    model_vam_false = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "vam", "pointer": None, "tanh_clipping": 50.0, "logit_bias": False, })
    
    path_am = "outputs/checkpoints/tsp100/am-am-None-100[]-multistart_sampling-0-False-0.0-v1/last.ckpt"
    path_vam = "outputs/checkpoints/tsp100/am-vam-None-100[]-multistart_sampling-0-True-0.0-v1/last.ckpt"
    path_vam_false = "outputs/checkpoints/tsp100/am-vam-None-100[]-multistart_sampling-0-False-0.0-v1/last.ckpt"
    model_am.load_state_dict(torch.load(path_am, weights_only=False)['state_dict'])
    model_vam.load_state_dict(torch.load(path_vam, weights_only=False)['state_dict'])
    model_vam_false.load_state_dict(torch.load(path_vam_false, weights_only=False)['state_dict'])

    policy_am = model_am.policy.to(device)
    out_am = policy_am.forward(
        td_am.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc, 
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
        num_starts=num_loc,
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
        num_starts=num_loc,
        multistart=True,
    )
    reward = out_vam_false['reward']
    actions = out_vam_false['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone()).to(device)
    cost_vam_false = -get_reward_unnormalized(unnormalized_td, action)

    if optimal_cost is None:
        print(f'{instance},N/A,{cost_am:},{cost_vam:},N/A%,{cost_vam_false:},N/A%,')
    else:
        print(f'{instance},{optimal_cost:},{cost_am:},{(cost_am - optimal_cost) / optimal_cost * 100:.2f}%,{cost_vam:},{(cost_vam - optimal_cost) / optimal_cost * 100:.2f}%,{cost_vam_false:},{(cost_vam_false - optimal_cost) / optimal_cost * 100:.2f}%,')