from tqdm.auto import tqdm

import torch
import torch.nn as nn

from .model import Generator, Discriminator, get_noise
from .dataset import get_dataloader
from .losses import get_disc_loss, get_gen_loss
from .utils import show_tensor_images, weights_init


def run_DCGAN():
    
    # Initiate hyper-parameters
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 50
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.0002
    beta_1 = 0.5 
    beta_2 = 0.999
    shuffle = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data
    dataloader = get_dataloader(batch_size=batch_size, shuffle=shuffle)
    
    # Initiate model
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
    
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    for epoch in range(n_epochs):
        print(f"\n ==== Epoch {epoch+1} ===== \n")
        
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            current_batch_size = len(real)
            real = real.to(device)

            ## Update discriminator ##
            
            # Zero out the gradients before backpropagation
            disc_opt.zero_grad()
            
            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, current_batch_size, z_dim, device)

            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            
            # Update gradients
            disc_loss.backward(retain_graph=True)
            
            # Update optimizer
            disc_opt.step()

            ## Update generator ##
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, current_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ## Visualization code ##
            if cur_step % display_step == 0 and cur_step > 0:
                print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(current_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(fake, current_step=cur_step, real=False)
                show_tensor_images(real, current_step=cur_step, real=True)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
    
    