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

def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    gen_block = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=output_dim),
        nn.BatchNorm1d(num_features=output_dim),
        nn.ReLU(inplace=True)
    )
    return gen_block
    
    
class Generator(nn.Module):
    '''
    Generator Class
    Params:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 1 x 28 x 28 = 784)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential([
            ('gen_block_1', get_generator_block(input_dim=z_dim, output_dim=hidden_dim)),
            ('gen_block_2', get_generator_block(input_dim=hidden_dim, output_dim=hidden_dim*2)),
            ('gen_block_3', get_generator_block(input_dim=hidden_dim*2, output_dim=hidden_dim*4)),
            ('gen_block_4', get_generator_block(input_dim=hidden_dim*4, output_dim=hidden_dim*8)),
            ('gen_block_5', nn.Linear(in_features=hidden_dim*8, out_features=im_dim)),
            nn.Sigmoid()
        ])
    
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        generated_image = self.gen(noise)
        return generated_image
        

# ========================================================================
# DISCRIMINATOR
# ========================================================================

def get_discriminator_block(input_dim, output_dim, negative_slope=0.2):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
    '''
    disc_block = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=output_dim),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    )
    return disc_block


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Params:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 1x28x28 = 784)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        
        self.disc = nn.Sequential([
            ('disc_block_1', get_discriminator_block(input_dim=im_dim, output_dim=hidden_dim*4)),
            ('disc_block_1', get_discriminator_block(input_dim=hidden_dim*4, output_dim=hidden_dim*2)),
            ('disc_block_1', get_discriminator_block(input_dim=hidden_dim*2, output_dim=hidden_dim)),
            nn.Linear(in_features=hidden_dim, out_features=1) # classify fake or real
        ])
        
    def foward(self, generated_image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        classification = self.disc(generated_image)
        return classification
    


        
        
    