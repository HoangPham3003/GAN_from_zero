import torch
import torch.nn as nn

# ========================================================================
# NOISE
# ========================================================================
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    
    noise_vector = torch.randn((n_samples, z_dim), device=device)
    return noise_vector


# ========================================================================
# GENERATOR
# ========================================================================

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_channels=z_dim, 
                                output_channels=hidden_dim*4),
            self.make_gen_block(input_channels=hidden_dim*4, 
                                output_channels=hidden_dim*2, 
                                kernel_size=4, 
                                stride=1)
            self.make_gen_block(input_channels=hidden_dim*2,
                                output_channels=hidden_dim)
            self.make_gen_block(input_channels=hidden_dim,
                                output_channels=im_chan,
                                kernel_size=4,
                                final_layer=True)
        )
        
    
    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels, 
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride),
                nn.BatchNorm2d(num_features=output_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels,
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride)
                nn.Tanh()
            )
            
    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)
    
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


# ========================================================================
# DISCRIMINATOR
# ========================================================================

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    '''
    
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential(
            self.make_disc_block(input_channels=im_chan, output_channels=hidden_dim),
            self.make_disc_block(input_channels=hidden_dim, output_channels=hidden_dim*2),
            self.make_disc_block(input_channels=hidden_dim*2, output_channels=1, final_layer=True)
        )
        
    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels, 
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=stride),
                nn.BatchNorm2d(num_features=output_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels, 
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=stride),
            )
            
    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_block), -1)