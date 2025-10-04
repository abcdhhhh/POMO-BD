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


# Load the problem from TSPLib
tsplib_dir = './tsplib'# modify this to the directory of your prepared files
files = os.listdir(tsplib_dir)
problem_files_full = [file for file in files if file.endswith('.tsp')]

# Load the optimal solution files from TSPLib
solution_files = [file for file in files if file.endswith('.opt.tour')] 

problems = []
# Load only problems with solution files
for sol_file in solution_files:
    prob_file = sol_file.replace('.opt.tour', '.tsp')
    problem = load_problem(os.path.join(tsplib_dir, prob_file))

    # NOTE: in some problem files (e.g. hk48), the node coordinates are not available
    # we temporarily skip these problems
    if not len(problem.node_coords):
        continue
    
    node_coords = torch.tensor([v for v in problem.node_coords.values()])
    solution = load_solution(os.path.join(tsplib_dir, sol_file))
    
    problems.append({
        "name": sol_file.replace('.opt.tour', ''),
        "node_coords": node_coords,
        "solution": solution.tours[0],
        "dimension": problem.dimension
    })
    
    
# order by dimension
problems = sorted(problems, key=lambda x: x['dimension'])

# Utils function: we will normalize the coordinates of the VRP instances
def normalize_coord(coord:torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_scaled = (x - x_min) / (x_max - x_min) 
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled 

def tsplib_to_td(problem, normalize=True):
    coords = torch.tensor(problem['node_coords']).float()
    coords_norm = normalize_coord(coords) if normalize else coords
    td = TensorDict({
        'locs': coords_norm,
    })
    td = td[None] # add batch dimension, in this case just 1
    return td

for problem in problems:
    instance = problem['name']
    # Inference
    td = tsplib_to_td(problem)
    td_orginal = tsplib_to_td(problem, normalize=False)

    num_loc = td['locs'].shape[1]

    if num_loc < 100 or num_loc > 300:
        continue

    env = TSPEnv(generator_params={"num_loc": num_loc})

    device = "cpu"

    bks_sol = (torch.tensor(problem['solution'], device=device, dtype=torch.int64) - 1)[None]
    unnormalized_td = env.reset(td_orginal.clone())
    optimal_cost = -env.get_reward(unnormalized_td, bks_sol).int().item()
    
    td_am = env.reset(td=td.clone()).to(device)
    td_scam = env.reset(td=td.clone()).to(device)

    # augment = CompositeAugmentation(
    #     num_augment_list=[8],
    #     augment_fn_list=['dihedral8'],
    # )

    # td_am = augment(td_am)
    # td_scam = augment(td_scam)

    model_am = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "am", "pointer": None, "context_k": 0, "tanh_clipping": 50.0, })
    model_scam = ReSym(env=env, policy_kwargs={"encoder": "am", "decoder": "scam", "pointer": None, "context_k": 1, "tanh_clipping": 50.0, })
    
    path_am = "outputs/checkpoints/tsp100/am-am-None-100[]-multistart_sampling-0-False-0.05/last.ckpt"
    path_scam = "outputs/checkpoints/tsp100/am-scam-None-100[]-multistart_sampling-1-False-0.05/last.ckpt"
    model_am.load_state_dict(torch.load(path_am, weights_only=False)['state_dict'])
    model_scam.load_state_dict(torch.load(path_scam, weights_only=False)['state_dict'])

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
    unnormalized_td = env.reset(td_orginal.clone())
    cost_am = -env.get_reward(unnormalized_td, action).int().item()

    policy_scam = model_scam.policy.to(device)
    out_scam = policy_scam.forward(
        td_scam.clone(), 
        env, 
        phase="test", 
        decode_type="multistart_greedy", 
        num_starts=num_loc, 
        multistart=True,
    )
    reward = out_scam['reward']
    actions = out_scam['actions']
    max_reward, max_idxs = reward.max(dim=-1)
    action = actions[max_idxs][None, :]
    unnormalized_td = env.reset(td_orginal.clone())
    cost_scam = -env.get_reward(unnormalized_td, action).int().item()

    print(f'{instance},{optimal_cost:},{cost_am:},{(cost_am - optimal_cost) / optimal_cost * 100:.2f}%,{cost_scam:},{(cost_scam - optimal_cost) / optimal_cost * 100:.2f}%,')