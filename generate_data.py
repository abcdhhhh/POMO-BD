import argparse
import logging
import os
import sys

from typing import List, Union

import numpy as np

from rl4co.data.utils import check_extension
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

from rl4co.data.generate_data import generate_env_data, DISTRIBUTIONS_PER_PROBLEM
from tqdm import tqdm

def get_vehicle_capacity(num_loc: int) -> int:
    """Capacity should be 30 + num_loc/5 if num_loc > 20 as described in Liu et al. 2024 (POMO-MTL).
    For every N over 1000, we add 1 of capacity every 33.3 nodes to align with Ye et al. 2024 (GLOP),
    i.e. 260 at 2K nodes, 350 at 5K nodes and 500 at 10K nodes.
    Note that this serves as a demand scaler.
    """
    if num_loc > 1000:
        extra_cap = 1000 // 5 + (num_loc - 1000) // 33.3
    elif num_loc > 20:
        extra_cap = num_loc // 5
    else:
        extra_cap = 0
    return 30 + extra_cap

VARIANT_FEATURES = {
    "CVRP": {"O": False, "TW": False, "L": False, "B": False, "M": False},
    "OVRP": {"O": True, "TW": False, "L": False, "B": False, "M": False},
    "VRPB": {"O": False, "TW": False, "L": False, "B": True, "M": False},
    "VRPL": {"O": False, "TW": False, "L": True, "B": False, "M": False},
    "VRPTW": {"O": False, "TW": True, "L": False, "B": False, "M": False},
    "OVRPTW": {"O": True, "TW": True, "L": False, "B": False, "M": False},
    "OVRPB": {"O": True, "TW": False, "L": False, "B": True, "M": False},
    "OVRPL": {"O": True, "TW": False, "L": True, "B": False, "M": False},
    "VRPBL": {"O": False, "TW": False, "L": True, "B": True, "M": False},
    "VRPBTW": {"O": False, "TW": True, "L": False, "B": True, "M": False},
    "VRPLTW": {"O": False, "TW": True, "L": True, "B": False, "M": False},
    "OVRPBL": {"O": True, "TW": False, "L": True, "B": True, "M": False},
    "OVRPBTW": {"O": True, "TW": True, "L": False, "B": True, "M": False},
    "OVRPLTW": {"O": True, "TW": True, "L": True, "B": False, "M": False},
    "VRPBLTW": {"O": False, "TW": True, "L": True, "B": True, "M": False},
    "OVRPBLTW": {"O": True, "TW": True, "L": True, "B": True, "M": False},
    "VRPMB": {"O": False, "TW": False, "L": False, "B": True, "M": True},
    "OVRPMB": {"O": True, "TW": False, "L": False, "B": True, "M": True},
    "VRPMBL": {"O": False, "TW": False, "L": True, "B": True, "M": True},
    "VRPMBTW": {"O": False, "TW": True, "L": False, "B": True, "M": True},
    "OVRPMBL": {"O": True, "TW": False, "L": True, "B": True, "M": True},
    "OVRPMBTW": {"O": True, "TW": True, "L": False, "B": True, "M": True},
    "VRPMBLTW": {"O": False, "TW": True, "L": True, "B": True, "M": True},
    "OVRPMBLTW": {"O": True, "TW": True, "L": True, "B": True, "M": True},
}

def generate_dataset_(
    filename: Union[str, List[str]] = None,
    data_dir: str = "data",
    name: str = None,
    problem: Union[str, List[str]] = "all",
    data_distribution: str = "all",
    dataset_size: int = 10000,
    graph_sizes: Union[int, List[int]] = [20, 50, 100],
    overwrite: bool = False,
    seed: int = 1234,
    disable_warning: bool = True,
    distributions_per_problem: Union[int, dict] = None,
    **kwargs,
):
    """We keep a similar structure as in Kool et al. 2019 but save and load the data as npz
    This is way faster and more memory efficient than pickle and also allows for easy transfer to TensorDict

    Args:
        filename: Filename to save the data to. If None, the data is saved to data_dir/problem/problem_graph_size_seed.npz. Defaults to None.
        data_dir: Directory to save the data to. Defaults to "data".
        name: Name of the dataset. Defaults to None.
        problem: Problem to generate data for. Defaults to "all".
        data_distribution: Data distribution to generate data for. Defaults to "all".
        dataset_size: Number of datasets to generate. Defaults to 10000.
        graph_sizes: Graph size to generate data for. Defaults to [20, 50, 100].
        overwrite: Whether to overwrite existing files. Defaults to False.
        seed: Random seed. Defaults to 1234.
        disable_warning: Whether to disable warnings. Defaults to True.
        distributions_per_problem: Number of distributions to generate per problem. Defaults to None.
    """

    if isinstance(problem, list) and len(problem) == 1:
        problem = problem[0]

    graph_sizes = [graph_sizes] if isinstance(graph_sizes, int) else graph_sizes

    if distributions_per_problem is None:
        distributions_per_problem = DISTRIBUTIONS_PER_PROBLEM

    if problem == "all":
        problems = distributions_per_problem
    elif problem in ["vrptw", "kp", "mtvrp"]:
        problems = {
            problem: [None]
            if data_distribution == "all"
            else [data_distribution]
        }
    else:
        problems = {
            problem: distributions_per_problem[problem]
            if data_distribution == "all"
            else [data_distribution]
        }

    # Support multiple filenames if necessary
    filenames = [filename] if isinstance(filename, str) else filename
    iter = 0

    # Main loop for data generation. We loop over all problems, distributions and sizes
    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in graph_sizes:
                if filename is None:
                    datadir = os.path.join(data_dir, problem, name)
                    os.makedirs(datadir, exist_ok=True)
                    fname = os.path.join(
                        datadir,
                        "{}{}.npz".format(
                            graph_size,
                            "_{}".format(distribution)
                            if distribution is not None
                            else "",
                        ),
                    )
                else:
                    try:
                        fname = filenames[iter]
                        # make directory if necessary
                        os.makedirs(os.path.dirname(fname), exist_ok=True)
                        iter += 1
                    except Exception:
                        raise ValueError(
                            "Number of filenames does not match number of problems"
                        )
                    fname = check_extension(filename, extension=".npz")

                if not overwrite and os.path.isfile(
                    check_extension(fname, extension=".npz")
                ):
                    if not disable_warning:
                        log.info(
                            "File {} already exists! Run with -f option to overwrite. Skipping...".format(
                                fname
                            )
                        )
                    continue

                # Set seed
                np.random.seed(seed)

                # Automatically generate dataset
                if problem == "vrptw":
                    dataset = generate_vrptw_data(
                        dataset_size, graph_size, capacities=kwargs.get("capacities"), max_time=3.0
                    )
                elif problem == "kp":
                    dataset = generate_kp_data(
                        dataset_size, graph_size, capacities=kwargs.get("capacities")
                    )
                elif problem == "mtvrp":
                    dataset = generate_mtvrp_data(
                        dataset_size,
                        num_loc=graph_size,
                        capacity=kwargs.get("capacity"),
                        min_demand=kwargs.get("min_demand", 1),
                        max_demand=kwargs.get("max_demand", 9),
                        scale_demand=kwargs.get("scale_demand", True),
                        max_time=kwargs.get("max_time", 4.6),
                        max_distance_limit=kwargs.get("max_distance_limit", 3.0),
                        speed=kwargs.get("speed", 1.0),
                        num_depots=kwargs.get("num_depots", 1),
                        variant=kwargs.get("variant", "CVRP"),
                    )
                else:
                    dataset = generate_env_data(
                        problem, dataset_size, graph_size, distribution, **kwargs
                    )

                # A function can return None in case of an error or a skip
                if dataset is not None:
                    # Save to disk as dict
                    log.info("Saving {} dataset to {}".format(problem, fname))
                    np.savez(fname, **dataset)

def generate_kp_data(dataset_size, kp_size, capacities=None):
    CAPACITIES = {
        50: 12.5,
        100: 25.0,
        200: 25.0,
    }

    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    if capacities is not None:
        for k, v in capacities.items():
            if k in CAPACITIES:
                print(f"Replacing capacity for {k} with {v}")
                CAPACITIES[k] = v

    return {
        "locs": np.random.uniform(size=(dataset_size, kp_size, 2)).astype(np.float32),
        "capacity": np.full(dataset_size, CAPACITIES[kp_size]).astype(np.float32),
    }

def generate_vrptw_data(dataset_size, vrp_size, capacities=None, max_time=3.0):
    # From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
    CAPACITIES = {
        10: 20.0,
        15: 25.0,
        20: 30.0,
        30: 33.0,
        40: 37.0,
        50: 40.0,
        60: 43.0,
        75: 45.0,
        100: 50.0,
        125: 55.0,
        150: 60.0,
        200: 70.0,
        500: 100.0,
        1000: 150.0,
    }

    # If capacities are provided, replace keys in CAPACITIES with provided values if they exist
    if capacities is not None:
        for k, v in capacities.items():
            if k in CAPACITIES:
                print(f"Replacing capacity for {k} with {v}")
                CAPACITIES[k] = v

    depot = np.random.uniform(size=(dataset_size, 2)).astype(
        np.float32
    )  # Depot location
    locs = np.random.uniform(size=(dataset_size, vrp_size, 2)).astype(
        np.float32
    )  # Node locations

    demand = np.random.randint(1, 10, size=(dataset_size, vrp_size)).astype(
        np.float32
    )  # Demand, uniform integer 1 ... 9
    
    capacity = np.full(dataset_size, CAPACITIES[vrp_size]).astype(np.float32)


    durations = np.full(shape=(dataset_size, vrp_size + 1), fill_value=0.2).astype(np.float32)
    durations[:, 0] = 0.0  # Depot duration is 0

    # 1. get distances from depot to locs
    dist = np.linalg.norm(
        depot[:, np.newaxis, :] - locs[:, :, :], axis=-1)
    dist = np.concatenate((np.zeros((dataset_size, 1)), dist), axis=1)

    #  2. define upper bound for time windows to make sure the vehicle can get back to the depot in time
    upper_bound = max_time - dist - durations

    # 3. time centers
    time_centers = np.random.uniform(
        size=(dataset_size, vrp_size + 1)
    ) * (upper_bound - dist) + dist

    # 4. time windows
    time_half_width = np.random.uniform(
        size=(dataset_size, vrp_size + 1)
    ) * (max_time / 3 - durations / 2) + durations / 2

    # 5. set min and max times
    min_times = np.clip(time_centers - time_half_width, dist, upper_bound)
    max_times = np.clip(time_centers + time_half_width, dist, upper_bound)

    # 6. reset times for depot
    min_times[..., :, 0] = 0.0
    max_times[..., :, 0] = max_time + 1e-5

    # 8. stack to tensor time_windows
    time_windows = np.stack((min_times, max_times), axis=-1).astype(np.float32)

    assert np.all(
        min_times < max_times
    ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."


    # Reset duration at depot to 0
    durations[:, 0] = 0.0

    return {
        "depot": depot,
        "locs": locs,
        "demand": demand,
        "capacity": capacity,
        "durations": durations,
        "time_windows": time_windows,
    }  # Capacity, same for whole dataset

def generate_mtvrp_data(
    dataset_size,
    num_loc=20,
    min_loc=0.0,
    max_loc=1.0,
    capacity=None,
    min_demand=1,
    max_demand=9,
    scale_demand=True,
    max_time=4.6,
    max_distance_limit=2.8,  # 2sqrt(2) ~= 2.8
    speed=1.0,
    num_depots=1,  # number of depots, only used for multi-depot problems
    variant="CVRP",
):
    """Generate MTVRP data using NumPy for a specific variant.
    NOTE: for MD (multi-depot) variants, we generate in the same way
    as single depot (considering the first one), but set the number of depots to the desired value.
    """
    # if first two letters are "md", strip and set multi-depot flag
    if variant[:2].lower() == "md":
        variant = variant[2:]
        num_depots = num_depots
    else:
        num_depots = 1  # single depot
    variant = variant.upper()
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"Unknown variant: {variant}")

    features = VARIANT_FEATURES[variant]

    if capacity is None:
        capacity = get_vehicle_capacity(num_loc)

    # Generate locations
    locs = np.random.uniform(min_loc, max_loc, (dataset_size, num_depots + num_loc, 2))

    # Generate demands
    def generate_demand(size):
        return (
            np.random.randint(min_demand, max_demand + 1, size).astype(np.float32)
            / capacity
        )

    demand_linehaul = generate_demand((dataset_size, num_loc))
    demand_backhaul = None

    if features["B"]:
        demand_backhaul = np.zeros((dataset_size, num_loc))
        backhaul_mask = (
            np.random.rand(dataset_size, num_loc) < 0.2
        )  # 20% of nodes are backhaul
        demand_backhaul[backhaul_mask] = generate_demand(backhaul_mask.sum())
        demand_linehaul[backhaul_mask] = 0

    # Generate backhaul class
    backhaul_class = (
        np.full((dataset_size, 1), 2 if features["M"] else 1) if features["B"] else None
    )

    # Generate open route
    open_route = np.full((dataset_size, 1), features["O"]) if features["O"] else None

    # Generate time windows and service time
    time_windows = None
    service_time = None
    if features["TW"]:
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * np.random.rand(dataset_size, num_loc)
        tw_length = b + (c - b) * np.random.rand(dataset_size, num_loc)
        # note: we assume that the first depot only for this
        d_0i = np.linalg.norm(locs[:, 0:1] - locs[:, num_depots:], axis=2)
        h_max = (max_time - service_time - tw_length) / d_0i * speed - 1
        tw_start = (
            (1 + (h_max - 1) * np.random.rand(dataset_size, num_loc)) * d_0i / speed
        )
        tw_end = tw_start + tw_length

        time_windows = np.concatenate(
            [
                np.zeros((dataset_size, num_depots, 2)),
                np.stack([tw_start, tw_end], axis=-1),
            ],
            axis=1,
        )
        time_windows[:, :num_depots, 1] = max_time
        # pad service time until the number of depots
        service_time = np.pad(service_time, ((0, 0), (num_depots, 0)))

    # Generate distance limits: dist_lower_bound = 2 * max(depot_to_location_distance),
    # max = min(dist_lower_bound, max_distance_limit). Ensures feasible yet challenging
    # constraints, with each instance having a unique, meaningful limit.
    if features["L"]:
        # Calculate the maximum distance from depot to any location
        max_dist = np.max(np.linalg.norm(locs[:, 1:] - locs[:, 0:1], axis=2), axis=1)

        # Calculate the minimum distance limit (2 * max_distance)
        distance_lower_bound = 2 * max_dist + 1e-6  # Add epsilon to avoid zero distance

        # Ensure max_distance_limit is not exceeded
        max_distance_limit = np.maximum(max_distance_limit, distance_lower_bound + 1e-6)

        # Generate distance limits between min_distance_limits and max_distance_limit
        distance_limit = np.random.uniform(
            distance_lower_bound,
            np.full_like(distance_lower_bound, max_distance_limit),
            (dataset_size,),
        )[:, None]
    else:
        distance_limit = None

    # Generate speed
    speed = np.full((dataset_size, 1), speed)

    # Scale demand if needed
    if scale_demand:
        vehicle_capacity = np.full((dataset_size, 1), 1.0)
    else:
        vehicle_capacity = np.full((dataset_size, 1), capacity)
        if demand_backhaul is not None:
            demand_backhaul *= capacity
        demand_linehaul *= capacity

    data = {
        "locs": locs.astype(np.float32),
        "demand_linehaul": demand_linehaul.astype(np.float32),
        "vehicle_capacity": vehicle_capacity.astype(np.float32),
        "speed": speed.astype(np.float32),
        "num_depots": np.full((dataset_size, 1), num_depots).astype(np.int32),
    }

    # Only include features that are used in the variant
    if features["B"]:
        data["demand_backhaul"] = demand_backhaul.astype(np.float32)
        data["backhaul_class"] = backhaul_class.astype(np.float32)
    if features["O"]:
        data["open_route"] = open_route
    if features["TW"]:
        data["time_windows"] = time_windows.astype(np.float32)
        data["service_time"] = service_time.astype(np.float32)
    if features["L"]:
        data["distance_limit"] = distance_limit.astype(np.float32)

    return data

if __name__ == '__main__':
    distributions = [("vrptw", "all"), ("kp", "all"), ("vrp", "all"), ("op", "all"), ("pctsp", "all"), ("tsp", "all"), ("pdp", "all")] #, ("atsp", "all")]
    for problem, data_distribution in tqdm(distributions):
        # set sizes  
        if problem == "tsp":
            sizes = [(10000, [20, 50, 100]), (1000, [200]), (128, [500, 1000, 2000, 5000]), (16, [10000])]
        elif problem == "vrp":
            sizes = [(10000, [20, 50, 100]), (1000, [200]), (128, [500]), (100, [1000])]
        elif problem == "op":
            sizes = [(10000, [20, 50, 100]), (128, [500, 1000])]
        elif problem == "pctsp":
            sizes = [(10000, [20, 50, 100]), (128, [500, 1000])]
        elif problem == "pdp":
            sizes = [(10000, [20, 50, 100]), (128, [500])]
        elif problem == "atsp":
            sizes = [(10000, [20, 50, 100]), (128, [500]), (16, [1000])]
        elif problem == "vrptw":
            sizes = [(10000, [20, 50, 100])]
        elif problem == "kp":
            sizes = [(10000, [50, 100, 200])]

        kwargs = {}
        # set max_lengths
        if problem == "op":
            kwargs['max_lengths'] = {20: 2.0, 50: 3.0, 100: 4.0, 200: 5.0, 500: 8.0, 1000: 12.0} # ReEvo, BQ
        elif problem == "pctsp":
            kwargs['max_lengths'] = {20: 2.0, 50: 3.0, 100: 4.0, 500: 9.0, 1000: 12.0, 2000: 15.0, 5000: 20.0} # GLOP, UDC

        for dataset_size, graph_sizes in sizes:
            for name, seed in [("val", 4321), ("test", 1234)]:
                generate_dataset_(
                    data_dir="./data",
                    name=name,
                    seed=seed,
                    problem=problem,
                    data_distribution=data_distribution,
                    dataset_size=dataset_size,
                    graph_sizes=graph_sizes,
                    **kwargs,
                )

    variants = list(VARIANT_FEATURES.keys())
    for problem in tqdm(variants):
        problem = problem.lower()
        for name, seed in [("val", 4321), ("test", 1234)]:
            for graph_size in [50, 100]:
                generate_dataset_(
                    data_dir="./data",
                    filename=f"./data/mtvrp/{problem}/{name}/{graph_size}.npz",
                    name=name,
                    seed=seed,
                    problem="mtvrp",
                    data_distribution="all",
                    dataset_size=1000,
                    graph_sizes=[graph_size],
                    variant=problem,
                )
