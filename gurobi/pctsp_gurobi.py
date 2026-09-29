#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import itertools
import numpy as np
import gurobipy as gp
import torch
from rl4co.envs.routing.pctsp.env import PCTSPEnv
from tqdm import tqdm

SEED = 42
num_loc = 20
max_penalty = 2.0


def pctsp_gurobi(data, timeout=None, start=None, visit=None, verbose=False):
    """
    Solves the Euclidan pctsp problem to pctsptimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """

    depot = data["depot"].tolist()
    loc = data["locs"].tolist()
    penalty = data["penalty"].tolist()
    prize = data["deterministic_prize"].tolist()
    min_prize = 1.

    points = [depot] + loc
    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if tour is not None:
                # add subtour elimination constraint for every pair of cities in tour
                # model.cbLazy(quicksum(model._vars[i, j]
                #                       for i, j in itertools.combinations(tour, 2))
                #              <= len(tour) - 1)

                model.cbLazy(gp.quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= gp.quicksum(model._dvars[i] for i in tour) * (len(tour) - 1) / float(len(tour)))

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges, exclude_depot=True):
        unvisited = list(range(n))
        #cycle = range(n + 1)  # initial length has 1 more city
        cycle = None
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            # If we do not yet have a cycle or this is the shorter cycle, keep this cycle
            # Unless it contains the depot while we do not want the depot
            if (
                (cycle is None or len(cycle) > len(thiscycle))
                    and len(thiscycle) > 1 and not (0 in thiscycle and exclude_depot)
            ):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        np.linalg.norm(np.array(points[i]) - np.array(points[j]))
        for i in range(n) for j in range(i)}

    model = gp.Model()

    # Create variables
    vars = model.addVars(dist.keys(), obj=dist, vtype=gp.GRB.BINARY, name='e')
    for i,j in dist.keys():
        vars[j,i] = vars[i,j] # edge in pctspposite direction

    # Depot vars can be 2
    for i,j in vars.keys():
        if i == 0 or j == 0:
            vars[i,j].vtype = gp.GRB.INTEGER
            vars[i,j].ub = 2

    penalty_dict = {
        i + 1: -p  # We get penalties for the nodes not visited so we 'save the penalties' for the nodes visited
        for i, p in enumerate(penalty)
    }
    delta = model.addVars(range(1, n), obj=penalty_dict, vtype=gp.GRB.BINARY, name='delta')

    # Add degree-2 constraint (2 * delta for nodes which are not the depot)
    model.addConstrs(vars.sum(i,'*') == (2 if i == 0 else 2 * delta[i]) for i in range(n))


    # Minimum prize constraint
    assert min_prize <= sum(prize)
    # Subtract 1 from i since prizes are 0 indexed while delta vars start with 1 (0 is depot)
    model.addConstr(gp.quicksum(var * prize[i - 1] for i, var in delta.items()) >= min_prize)

    # start
    if start is not None:
        model.addConstr(vars[0, start] == 1)

    if visit is not None:
        model.addConstr(delta[visit] == 1)

    # optimize model
    model._vars = vars
    model._dvars = delta

    # Set parameters
    if timeout is not None:
        model.Params.timeLimit = timeout
    model.Params.outputFlag = verbose
    model.Params.lazyConstraints = 1
    model.Params.threads = 1 # Keep 1 thread for callback
    model.Params.seed = SEED

    # For the correct objective, we need to add the sum of the penalties, which are subtracted when nodes are visited
    # this is important for the relative gap termination criterion
    model.objcon = sum(penalty)
    model.optimize(subtourelim)

    vals = model.getAttr('x', vars)
    selected = gp.tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected, exclude_depot=False)
    assert tour[0] == 0, "Tour should start with depot"

    return tour, model.objVal

if __name__ == "__main__":
    torch.manual_seed(SEED)
    env = PCTSPEnv(generator_params={"num_loc": num_loc, "max_penalty": max_penalty})
    device = "cpu"
    data = env.generator._generate(batch_size=[1]).to(device)

    rewards = []
    for i, d in tqdm(enumerate(data)):
        vals = []

        # for start in range(1, num_loc + 1):
        #     tour, obj_val = pctsp_gurobi(d, start=start, verbose=False)

        #     td = env._reset(data[i:i+1], batch_size=(1,))
        #     env.check_solution_validity(td, torch.tensor(tour).unsqueeze(0).to(device))

        #     # print(start, tour, obj_val)
        #     vals.append(obj_val)

        for visit in range(1, num_loc + 1):
            tour, obj_val = pctsp_gurobi(d, visit=visit, verbose=False)

            td = env._reset(data[i:i+1], batch_size=(1,))
            env.check_solution_validity(td, torch.tensor(tour).unsqueeze(0).to(device))

            # print(visit, tour, obj_val)
            vals.append(obj_val)
        vals.append(min(vals))

        info = [min(vals), max(vals), sum(vals) / len(vals)]
        # print(info)
        rewards.append(info)

    rewards = np.array(rewards)
    print("Min:", rewards[:,0].mean(), "Max:", rewards[:,1].mean(), "Avg:", rewards[:,2].mean())