from transform_datasets.dataset import TransformDataset


def gen_dataset(config, return_tlabels=False):
    dataset = config["dataset"]["type"](**config["dataset"]["params"])
    transforms = [v["type"](**v["params"]) for k, v in config["transforms"].items()]
    t_dataset = TransformDataset(dataset, transforms, return_tlabels)
    return t_dataset
