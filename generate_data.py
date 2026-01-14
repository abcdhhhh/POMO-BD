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
    elif problem in ["vrptw", "kp"]:
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
                    datadir = os.path.join(data_dir, problem)
                    os.makedirs(datadir, exist_ok=True)
                    fname = os.path.join(
                        datadir,
                        "{}{}{}_{}_seed{}.npz".format(
                            problem,
                            "_{}".format(distribution)
                            if distribution is not None
                            else "",
                            graph_size,
                            name,
                            seed,
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


if __name__ == '__main__':
    for problem, data_distribution in [("vrptw", "all"), ("kp", "all"), ("vrp", "all"), ("op", "dist"), ("pctsp", "all"), ("tsp", "all"), ("pdp", "all")]: #, ("atsp", "all")]:
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
