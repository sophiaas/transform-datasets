import wandb
from transform_datasets.utils.wandb import create_dataset
from transform_datasets.utils.core import flatten_dict, nest_dict
from transform_datasets.patterns.synthetic import *
from transform_datasets.transforms import *


PROJECT = "datasets"
ENTITY = "naturalcomputation"

dataset = {
    "type": HarmonicsS1,
    "params": {"dim": 256, "n_classes": 10, "seed": 0},
}

transforms = {
    "0": {
        "type": CyclicTranslation1D,
        "params": {"fraction_transforms": 1.0, "sample_method": "linspace"},
    },
    "1": {"type": UniformNoise, "params": {"n_samples": 1, "magnitude": 0.5}},
}


config_dict = {"dataset": dataset, "transforms": transforms}

config = flatten_dict(config_dict)

sweep_config = {
    "method": "grid",
    "parameters": {
        "dataset.params.n_classes": {"values": [10, 100, 1000, 2000, 3000, 4000, 5000]},
        "transforms.1.params.magnitude": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
    },
}


def dataset_sweep():
    with wandb.init(config=config, job_type="dataset_sweep") as run:
        new_config = dict(wandb.config)
        new_config = nest_dict(new_config)

        new_config["dataset"]["type"] = config_dict["dataset"]["type"]
        for k in config_dict["transforms"]:
            new_config["transforms"][k]["type"] = config_dict["transforms"][k]["type"]

        dataset = load_or_create_dataset(new_config, PROJECT, ENTITY, run)


sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)

wandb.agent(
    sweep_id,
    function=dataset_sweep,
)
