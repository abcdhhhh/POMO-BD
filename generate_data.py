from rl4co.data.generate_data import generate_dataset

generate_dataset(
    data_dir="./data",
    name="val",
    seed=4321,
    problem="pctsp",
    dataset_size=128,
    graph_sizes=[500, 1000],
)