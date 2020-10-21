import argparse


class Settings:
    def __init__(self):
        self.initialized = False

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='One-Shot Learning running script')
        parser.add_argument('--dataroot', type=str, default="small_train_mnist.npz",
                            help='path to labeled train dataset')
        parser.add_argument('--dataroot_unlabeled', type=str, default="small_train_mnist_unlabeled.npz",
                            help='path to unlabeled train dataset')
        parser.add_argument('--dataroot_test', type=str, default="small_test_mnist.npz",
                            help='path to test dataset ')
        parser.add_argument('--lr', type=float, default=0.003,
                            help='learning rate')
        parser.add_argument('--embedding_size', type=int, default=10,
                            help='classifier output')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--n_samples', type=int, default=5000,
                            help='number of samples extracted from labeled data')
        parser.add_argument('--epochs', type=int, default=50,
                            help='number of epochs')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help='first beta for Adam optimizer')
        parser.add_argument('--stage', type=str, default="Train",
                            help="'Test' omits optimizers")
        parser.add_argument('--lr_decay_epoch', type=int, default=25,
                            help='epoch lr starts decaying')
        parser.add_argument('--checkpointroot_save', type=str, default="./checkpoints/",
                            help='epoch lr starts decaying')
        parser.add_argument('--checkpointroot_load', type=str, default="./checkpoints/Semi_Model2.pth")
        settings = parser.parse_args()
        self.initialized = True
        return settings

