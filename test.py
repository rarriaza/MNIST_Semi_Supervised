from collections import OrderedDict
import torch
from tqdm import tqdm

from settings import Settings
from dataset import Dataset
from network import EmbeddingNet


def load_model(settings):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = EmbeddingNet(settings).to(device)
    model = load_parameters_model(settings, model)
    model.eval()
    return model


def load_parameters_model(settings, model):
    if torch.cuda.is_available():
        model.load_state_dict(
            torch.load(settings.checkpointroot_load)["state_dict"])
    else:
        model = load_from_cpu(settings, model)
    return model


def get_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        parts = k.split(".")
        name = ""
        for p in parts:
            if p != "module":
                name += p + "."
        name = name[0:-1]
        new_state_dict[name] = v
    return new_state_dict


def load_from_cpu(settings, model):
    state_dict = torch.load(settings.checkpointroot_load, map_location=torch.device('cpu'))["state_dict"]
    params = get_state_dict(state_dict)
    model.load_state_dict(params)
    return model


if __name__ == '__main__':
    settings = Settings().get_defaults()
    dataloader = Dataset(settings).create_dataset(augmentation=False, unlabeled=False, test=True)
    model = load_model(settings)
    batchsize = settings.batch_size
    n_batches = len(dataloader.dataset.data["labels"]) // batchsize
    accuracy = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
            model.set_input(data)
            model.forward()
            accuracy += model.compute_accuracy() / (n_batches * batchsize)
    print("final accuracy: {}".format(accuracy))

