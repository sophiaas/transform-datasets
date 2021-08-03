from transform_datasets.dataset import TransformDataset
import jsonpickle
import hashlib
import pandas as pd
import wandb
import torch
import os


def nest_dict(dict1):
    result = {}
    for k, v in dict1.items():

        # for each key call method split_rec which
        # will split keys to form recursively
        # nested dictionary
        split_rec(k, v, result)
    return result


def split_rec(k, v, out, sep="."):

    # splitting keys in dict
    # calling_recursively to break items on '_'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def flatten_dict(dd, separator=".", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def config_to_hash(config):
    if type(config) != dict:
        config = dict(config)
    flat_config = pd.json_normalize(config).to_dict()
    flat_config = sorted(flat_config.items())
    config_hash = hashlib.md5(jsonpickle.encode(flat_config).encode("utf-8")).digest()
    return config_hash.hex()


def gen_dataset(config):
    """
    Generate a TransformDataset from a config dictionary with the following
    structure:

    config = {
        "dataset": {"type": obj, "params": {}},
        "transforms": {
            "0": {"type": obj, "params": {}},
            "1": {"type": obj, "params": {}}
         }
    }

    The "type" parameter in each dictionary specifies an uninstantiated dataset
    or transform class. The "params" parameter specifies a dictionary containing
    the keyword arguments needed to instantiate the class.
    """
    dataset = config["dataset"]["type"](**config["dataset"]["params"])
    transforms = [
        config["transforms"][k]["type"](**config["transforms"][k]["params"])
        for k in sorted(config["transforms"])
    ]
    t_dataset = TransformDataset(dataset, transforms)
    return t_dataset


#
#
# def td_config_to_params(config):
#     config["dataset"] = config_to_params(config["dataset"])
#     for t in config["transforms"]:
#         config["transforms"][t] = config_to_params(config["transforms"][t])
#     return config
