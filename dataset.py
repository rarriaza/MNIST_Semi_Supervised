import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
from skimage.morphology import dilation, erosion


class Dataset:
    def __init__(self, settings):
        self.settings = settings
        self.images_path = settings.dataroot
        self.batch_size = settings.batch_size

    def create_dataset(self, augmentation=True, unlabeled=False, test=False):
        dataloader = SingleDataLoader(self.settings, unlabeled=unlabeled, augmentation=augmentation, test=test)
        dataset = dataloader.load_data()
        return dataset


class SingleDataset(data.Dataset):
    def __init__(self, settings, unlabeled=False, augmentation=True, test=False):
        self.settings = settings
        self.dataroot = settings.dataroot
        self.dataroot_unlabeled = settings.dataroot_unlabeled
        self.dataroot_test = settings.dataroot_test
        self.augmentation = augmentation
        self.set_random_augmentation()
        self.load_traindata(unlabeled=unlabeled, test=test)

    def load_traindata(self, unlabeled=False, test=False):
        if test:
            self.data = np.load(self.dataroot_test, allow_pickle=True)["arr_0"].item()
        elif unlabeled:
            self.data = np.load(self.dataroot_unlabeled, allow_pickle=True)["arr_0"].item()
        else:
            self.data = np.load(self.dataroot, allow_pickle=True)["arr_0"].item()

    def __getitem__(self, index):
        if self.augmentation:
            index = random.sample(range(len(self.data["labels"])), 1)[0]
        image, label = self.data["images"][index, :, :], torch.from_numpy(np.array(self.data["labels"][index]))
        if self.augmentation:
            image_tensor = self.transforms(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        return {"images": image_tensor, "labels": label, "indexes": index}

    def __len__(self):
        if self.augmentation:
            return self.settings.n_samples
        return len(self.data["labels"])

    def set_random_augmentation(self):
        self.transforms = transforms.Compose([
            RandomMorphology(),
            Image.fromarray,
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=(-10, 10, -10, 10)),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])


class SingleDataLoader():
    def __init__(self, settings, unlabeled=False, augmentation=True, test=False):
        self.settings = settings
        self.images_path = settings.dataroot
        self.batch_size = settings.batch_size
        self.dataset = SingleDataset(settings, unlabeled=unlabeled, augmentation=augmentation, test=test)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0, drop_last=True)

    def load_data(self):
        return self

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data


class RandomMorphology:
    def __call__(self, image):
        transform = random.choice(["dilation", "erosion", "None"])
        if transform == "dilation":
            return dilation(image)
        elif transform == "erosion":
            return erosion(image)
        return image
