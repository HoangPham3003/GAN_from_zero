from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_dataloader(batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataloader = DataLoader(
        dataset=MNIST(root='./data',
              train=True,
              download=True,
              transform=transform),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return dataloader