from transform_datasets.dataset import TransformDataset


def gen_dataset(config, return_tlabels=False):
    """
    Generate a TransformDataset from a config dictionary with the following
    structure:

    config = {
        "dataset": {"type": obj, "params": {}},
        "transforms": {
            "transform1": {"type": obj, "params": {}},
            "transform2": {"type": obj, "params": {}}
         }
    }

    The "type" parameter in each dictionary specifies an uninstantiated dataset
    or transform class. The "params" parameter specifies a dictionary containing
    the keyword arguments needed to instantiate the class.
    """
    dataset = config["dataset"]["type"](**config["dataset"]["params"])
    transforms = [v["type"](**v["params"]) for k, v in config["transforms"].items()]
    t_dataset = TransformDataset(dataset, transforms, return_tlabels)
    return t_dataset
