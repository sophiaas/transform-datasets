from transform_datasets.patterns.synthetic import *
from transform_datasets.transforms import *


dataset = {
    "type": HarmonicsS1,
    "params": {"dim": 256, "n_classes": 10, "seed": 0},
}

transforms = {
    "translation": {
        "type": CyclicTranslation1D,
        "params": {"fraction_transforms": 1.0, "sample_method": "linspace"},
    },
    "noise": {"type": UniformNoise, "params": {"n_samples": 1, "magnitude": 0.5}},
}


config = {"dataset": dataset, "transforms": transforms}

sweep_config = {
    "method": "grid",
    "parameters": {
        "dataset.params.n_classes": {"values": [100, 1000, 10000]},
        "transforms.noise.params.magnitude": {
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
    },
}
