r"""
Gurobi Solver for CVRP
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

SEED = 42
num_loc = 100
vehicle_capacity = 50.0

import numpy as np
import gurobipy as gp
import torch
from rl4co.envs.routing.cvrp.env import CVRPEnv 
from tqdm import tqdm


def cvrp_gurobi(data, timeout=None, start=None, verbose=False):
    """
    Gurobi + Miller-Tucker-Zemlin
    """
    # Preparation
    locs = torch.cat([data["depot"].unsqueeze(-2), data["locs"]], dim=-2)
    dists = torch.cdist(locs, locs, p=2).cpu().numpy()
    nodes_num = len(dists) - 1
    norm_demands = data["demand"].cpu().numpy()
    
    # Create gurobi model
    model = gp.Model(f"CVRP-{nodes_num}-{data['capacity'].item()}")
    
    # Decision vars: x[i,j] = 1 if arc (i,j) is used
    x = model.addVars(
        range(nodes_num+1), range(nodes_num+1), vtype=gp.GRB.BINARY, name="x"
    )
    
    # MTZ variables for subtour elimination (load / flow formulation)
    u = model.addVars(
        range(1, nodes_num+1), 
        lb={i: norm_demands[i-1] for i in range(1, nodes_num+1)}, 
        ub=1.0, 
        name="u"
    )
    
    # Objective: minimize distance
    model.setObjective(
        expr=gp.quicksum(dists[i][j] * x[i,j] for i in range(nodes_num+1) \
            for j in range(nodes_num+1) if i != j), 
        sense=gp.GRB.MINIMIZE
    )
    
    # Constraints
    # 1. Each customer visited exactly once
    for j in range(1, nodes_num+1):
        model.addConstr(gp.quicksum(x[i,j] for i in range(nodes_num+1) if i != j) == 1)
    for i in range(1, nodes_num+1):
        model.addConstr(gp.quicksum(x[i,j] for j in range(nodes_num+1) if i != j) == 1)

    # 2. Vehicle flow conservation at depot
    # model.addConstr(gp.quicksum(x[0,j] for j in range(1, nodes_num+1)) >= 1)
    # model.addConstr(gp.quicksum(x[i,0] for i in range(1, nodes_num+1)) >= 1)
    
    # 3. MTZ subtour elimination with capacity
    for i in range(1, nodes_num+1):
        for j in range(1, nodes_num+1):
            if i != j:
                model.addConstr(u[i] - u[j] + 1.0 * x[i,j] <= 1.0 - norm_demands[j-1])

    # start
    if start is not None:
        model.addConstr(x[0, start] == 1)
     
    # Set parameters
    if timeout is not None:
        model.Params.timeLimit = timeout
    model.Params.outputFlag = verbose
    model.Params.lazyConstraints = 1
    model.Params.threads = 1 # Keep 1 thread for callback
    model.Params.seed = SEED

    # Optimize
    model.optimize()

    # Get & Store the solution
    try:
        adj_matrix = np.zeros(shape=(nodes_num+1, nodes_num+1), dtype=np.int32)
        for i in range(nodes_num+1):
            for j in range(nodes_num+1):
                if i == j:
                    continue
                adj_matrix[i][j] = round(x[i, j].X)
    except:
        raise Exception("No solution found")

    # Get the tour from the adjacency matrix
    tour = [0]
    cur_node = 0
    visited = np.zeros_like(norm_demands).astype(np.bool_)
    while not visited.all():
        next_nodes = np.nonzero(adj_matrix[cur_node])[0]
        next_node = next_nodes[0]
        tour.append(next_node)
        if cur_node == 0:
            adj_matrix[0][next_node] = 0
        else:
            visited[cur_node-1] = True
        cur_node = next_node

    tour = np.array(tour)
    return tour, model.ObjVal, model.Status

if __name__ == "__main__":
    torch.manual_seed(SEED)
    env = CVRPEnv(generator_params={"num_loc": num_loc, "vehicle_capacity": vehicle_capacity})
    device = "cpu"
    data = env.generator._generate(batch_size=[1]).to(device)

    tour, obj_val, status = cvrp_gurobi(data[0], timeout=3600., start=None, verbose=True)

    # rewards = []
    # for i, d in tqdm(enumerate(data)):
    #     vals = []
    #     for start in range(1, num_loc + 1):
    #         tour, obj_val, status = cvrp_gurobi(data[i], timeout=None, start=start, verbose=False)

    #         td = env._reset(data[i:i+1], batch_size=(1,))
    #         env.check_solution_validity(td, torch.tensor(tour).unsqueeze(0).to(device))

    #         # print(start, tour, obj_val, status)
    #         vals.append(obj_val)
    #     info = [min(vals), max(vals), sum(vals) / len(vals)]
    #     # print(info)
    #     rewards.append(info)
    
    # rewards = np.array(rewards)
    # print("Min:", rewards[:,0].mean(), "Max:", rewards[:,1].mean(), "Avg:", rewards[:,2].mean())