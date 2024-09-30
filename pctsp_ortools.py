
import numpy as np
import pickle
import math
from tqdm import tqdm
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from rl4co.envs import PCTSPEnv

def float_to_scaled_int(v):
    return int(v * 10000000 + 0.5)

def euclidian_distance(position_1, position_2):
  """Computes the Euclidian distance between two points"""
  return int(math.sqrt((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2) + 0.5)


def solve(depot, locs, penalty, prize, sec_local_search = 0):
    data = {}
    data["num_vehicles"] = 1
    data["depot"] = 0
    data["locs"] = [(float_to_scaled_int(l[0]), float_to_scaled_int(l[1])) for l in [depot] + locs]
    data["prize"] = [float_to_scaled_int(v) for v in prize]
    data["penalty"] = [float_to_scaled_int(v) for v in penalty]
    data["min_prize"] = min(float_to_scaled_int(1), sum(data["prize"]))
    data["distances"] = {}
    for i in range(len(data["locs"])):
        data["distances"][i] = {}
        for j in range(len(data["locs"])):
            if j == i:
                data["distances"][i][j] = 0
            else:
                data["distances"][i][j] = euclidian_distance(data["locs"][i], data["locs"][j])


    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["locs"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distances"][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def prize_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['prize'][from_node - 1] if from_node else 0
    prize_callback_index = routing.RegisterUnaryTransitCallback(prize_callback)
    dimension_name = 'Prize'
    routing.AddDimension(
        prize_callback_index,
        0,  
        sum(data["prize"]),  
        True,  
        dimension_name,
    )
    prize_dimension = routing.GetDimensionOrDie(dimension_name)

    index = routing.End(0)
    routing.solver().Add(prize_dimension.CumulVar(index) >= data['min_prize'])

    # allow to drop nodes
    nodes = [routing.AddDisjunction([int(c + 1)], p) for c, p in enumerate(data["penalty"])]

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if sec_local_search > 0:
        # Additionally do local search
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = sec_local_search

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route.append(node_index)
        index = solution.Value(routing.NextVar(index))
    return solution.ObjectiveValue() / 10000000, route

if __name__ == '__main__':
    fn = 'data/pctsp/pctsp500_test_seed1234.npz'
    ac = np.load(fn)
    locs, depot, penalty, deterministic_prize, stochastic_prize = [a.tolist() for a in [ac["locs"], ac["depot"], ac["penalty"], ac["deterministic_prize"], ac["stochastic_prize"]]]
    ac = list(zip(depot, locs, penalty, deterministic_prize, stochastic_prize))

    # fn = 'pctsp500_test_seed1234.pkl'
    # with open(fn, 'rb') as fp:
    #     ac = pickle.load(fp)

    obj_list = []
    for instance in tqdm(ac):
        depot, locs, penalty, deterministic_prize, stochastic_prize = instance
        obj, route = solve(depot, locs, penalty, deterministic_prize, 0)
        obj_list.append(obj)
        print(np.mean(obj_list))



