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
                dataset = generate_env_data(
                    problem, dataset_size, graph_size, distribution, **kwargs
                )

                # A function can return None in case of an error or a skip
                if dataset is not None:
                    # Save to disk as dict
                    log.info("Saving {} dataset to {}".format(problem, fname))
                    np.savez(fname, **dataset)

if __name__ == '__main__':
    for problem, data_distribution in [("tsp", "all"), ("vrp", "all"), ("op", "dist"), ("pctsp", "all")]:
        # set sizes  
        if problem == "tsp":
            sizes = [(10000, [20, 50, 100]), (128, [500, 1000, 2000, 5000]), (16, [10000])]
        elif problem == "vrp":
            sizes = [(10000, [20, 50, 100]), (128, [500]), (100, [1000, 2000, 5000, 7000])]
        elif problem in ["op", "pctsp"]:
            sizes = [(10000, [20, 50, 100]), (128, [500, 1000]), (16, [2000])]

        kwargs = {}
        # set capacities
        if problem == "vrp":
            kwargs['capacities'] = {
                20: 30.0,
                50: 40.0,
                100: 50.0,
                500: 100.0,
                1000: 200.0,
                2000: 300.0,
                5000: 300.0,
                7000: 300.0,
            }
        # set max_lengths
        if problem == "op":
            kwargs['max_lengths'] = {20: 2.0, 50: 3.0, 100: 4.0, 500: 4.0, 1000: 4.0, 2000: 4.0, 5000: 4.0}
        elif problem == "pctsp":
            kwargs['max_lengths'] = {20: 2.0, 50: 3.0, 100: 4.0, 500: 9.0, 1000: 12.0, 2000: 15.0, 5000: 20.0} 

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
