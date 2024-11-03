import torch

from model import get_noise


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    
    noise = get_noise(n_samples=num_images, z_dim=z_dim, device=device)
    fake_images = gen(noise).detach()  
    
    # Predict the fake -> want to reach 0
    fake_pred = disc(fake_images)
    fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
    
    # Predict the real -> want to reach 1
    real_pred = disc(real)
    real_loss = criterion(real_pred, torch.ones_like(real_pred))
    
    disc_loss = (fake_loss + real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    noise = get_noise(n_samples=num_images, z_dim=z_dim, device=device)
    fake_images = gen(noise)
    disc_pred = disc(fake_images)
    
    # Want the fake image look real -> prediction of disc reach 1
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss
    