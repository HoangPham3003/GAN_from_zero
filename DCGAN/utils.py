import pathlib
import matplotlib.pyplot as plt

import torch.nn as nn
from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), current_step=500, real=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    state = "fake"
    if real:
        state = "real"
    plt.axis('off')
    fig_name = "step_{}_{}".format(current_step, state)
    plt.title(fig_name)
    current_path = pathlib.Path(__file__).parent.resolve()
    plt.savefig(f"{current_path}/runs/{fig_name}.jpg")
    plt.show()
    
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

