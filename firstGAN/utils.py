import matplotlib.pyplot as plt

from torchvision.utils import make_grid


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), current_step=500):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrows=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.save_fig("runs/{}.jpg".format(current_step))
    plt.show()

