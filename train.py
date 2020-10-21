from network import EmbeddingNet
from settings import Settings
from dataset import Dataset

import torch
import numpy as np
from tqdm import tqdm, trange


def train(model, dataloader_labeled, dataloader_unlabeled, settings):
    batchsize = settings.batch_size
    epochs = settings.epochs
    train_loss = np.zeros(epochs)

    for epoch in trange(epochs, desc='Epoch: '):
        train_unlabeled(model, dataloader_unlabeled, batchsize)
        train_labeled(model, dataloader_labeled, batchsize, settings.n_samples)
        train_loss[epoch] += model.cum_loss
        model.cum_loss = 0.0
        model.lr_decay(epoch)

    save_model(model, settings)


def train_unlabeled(model, dataloader, batchsize):
    n_samples = dataloader.dataset.__len__()
    n_batches = n_samples // batchsize

    for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
        model.eval()
        with torch.no_grad():
            model.set_input(data)
            model.forward()
            model.get_fake_label()
        model.train()
        model.forward()
        model.backward_net(n_samples)


def train_labeled(model, dataloader, batchsize, n_samples):
    n_batches = n_samples // batchsize

    for i, data in enumerate(tqdm(dataloader, total=n_batches, desc="Batch: ", leave=False)):
        model.set_input(data)
        model.forward()
        model.backward_net(n_samples)


def create_dataloaders(args):
    dataloader_labeled = Dataset(args).create_dataset(augmentation=True, unlabeled=False)
    dataloader_unlabeled = Dataset(args).create_dataset(augmentation=False, unlabeled=True)
    return dataloader_labeled, dataloader_unlabeled


def init_model(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = EmbeddingNet(args).to(device)
    return model


def save_model(model, args):
    state = {
        'state_dict': model.state_dict()
    }
    torch.save(state, args.checkpointroot_save + "Semi_Model_bla.pth")


if __name__ == '__main__':
    settings = Settings().parse_arguments()
    model = init_model(settings)
    dataloader_labeled, dataloader_unlabeled = create_dataloaders(settings)
    train(model, dataloader_labeled, dataloader_unlabeled, settings)

