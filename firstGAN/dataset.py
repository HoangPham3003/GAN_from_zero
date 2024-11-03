from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_dataloader(batch_size, shuffle=True):
    dataloader = DataLoader(
        dataset=MNIST(root='./data',
              train=True,
              download=True,
              transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader