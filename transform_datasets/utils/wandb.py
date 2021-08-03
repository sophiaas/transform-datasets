from transform_datasets.utils import (
    config_to_hash,
    gen_dataset,
    flatten_dict,
)
import wandb
import os
import torch


def get_names(config, project, entity):
    dataset_type = config["dataset"]["type"].__name__
    dataset_hash = config_to_hash(config)

    transform_name = "-".join(
        [config["transforms"][k]["type"].__name__ for k in sorted(config["transforms"])]
    )
    dataset_name = dataset_type + "_" + transform_name
    return dataset_name, dataset_type, dataset_hash


def get_artifact(config, project, entity):
    dataset_name, dataset_type, dataset_hash = get_names(config, project, entity)
    path = "{}/{}/{}:{}".format(entity, project, dataset_name, dataset_hash)
    api = wandb.Api(overrides={"project": project, "entity": entity})
    artifact = api.artifact(
        path,
        type=dataset_type,
    )
    return artifact


def create_dataset(config, project, entity, run=None):

    try:
        artifact = get_artifact(config, project, entity)

        print(
            "Dataset has been previously generated and pushed to wandb at {}/{}/{}".format(
                entity, project, artifact._artifact_name
            )
        )
    except:
        if run is None:
            run = wandb.init(project=project, entity=entity, job_type="create_data")
            finish_run = True
        else:
            finish_run = False
        dataset_name, dataset_type, dataset_hash = get_names(config, project, entity)

        artifact = wandb.Artifact(
            name=dataset_name, type=dataset_type, metadata=dict(config)
        )

        dataset = gen_dataset(config)
        with artifact.new_file("dataset.pt", mode="wb") as file:
            torch.save(dataset, file)
        run.log_artifact(artifact)
        artifact.wait()
        artifact.aliases.append(dataset_hash)
        artifact.save()
        if finish_run:
            run.finish()


def load_dataset(config, project, entity):
    try:
        artifact = get_artifact(config, project, entity)
        path = artifact.download()
        dataset = torch.load(os.path.join(path, "dataset.pt"))
        return dataset
    except:
        raise FileNotFoundError


def load_or_create_dataset(config, project, entity):
    try:
        dataset = load_dataset(config, project, entity)
    except:
        create_dataset(config, project, entity)
        dataset = load_dataset(config, project, entity)
    return dataset
