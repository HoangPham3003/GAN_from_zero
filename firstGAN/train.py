from tqdm.auto import tqdm

import torch
import torch.nn as nn

from .model import Generator, Discriminator, get_noise
from .dataset import get_dataloader
from .losses import get_disc_loss, get_gen_loss
from .utils import show_tensor_images


def run_firstGAN():
    
    # Initiate hyper-parameters
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001
    shuffle = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data
    dataloader = get_dataloader(batch_size=batch_size, shuffle=shuffle)
    
    # Initiate model
    gen = Generator(z_dim=z_dim).to(device)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=lr)
    
    current_step = 0
    mean_generator_loss = 0.
    mean_discriminator_loss = 0.
    test_generator = True
    gen_loss = False
    error = False
    
    for epoch in range(n_epochs):
        print(f"\n ==== Epoch {epoch+1} ===== \n")
        
        # Dataloader returns the batches
        for real, _ in tqdm(dataloader):
            current_batch_size = len(real)
            
            # Flatten the batch of real images from the dataset
            real = real.view(current_batch_size, -1).to(device)
            
            ### Update discriminator ###
            # Zero out the gradients before backpropagation
            disc_optimizer.zero_grad()
            
            # Calculate discriminator loss
            disc_loss = get_disc_loss(gen, disc, criterion, real, current_batch_size, z_dim, device)
            
            # Compute gradients
            disc_loss.backward(retain_graph=True)
            
            # Update weights
            disc_optimizer.step()
            
            # For testing purposes, to keep track of the generator weights
            if test_generator:
                old_generator_weights = gen.gen[0][0].weight.detach().clone()

            ### Update generator ###
            gen_optimizer.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, current_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_optimizer.step()
            
            if test_generator:
                try:
                    assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                    assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                except:
                    error = True
                    print("Runtime tests have failed")  
                    
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code ###
            if current_step % display_step == 0 and current_step > 0:
                print(f"Step {current_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(current_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_tensor_images(image_tensor=fake, current_step=current_step, real=True)
                show_tensor_images(image_tensor=real, current_step=current_step, real=False)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            current_step += 1
            
    

    