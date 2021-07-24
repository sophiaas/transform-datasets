from transform_datasets.patterns.synthetic import *
from transform_datasets.transforms import *


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


config = {"dataset": dataset, "transforms": transforms}
