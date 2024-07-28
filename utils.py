import os
import torch
import torch
import torch.nn.functional as F
from PIL import Image
import os
from model import VAE


def loss_criterion(recon_x, x, logvar, mu):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = VAE(input_size=checkpoint['model_state_dict']['fc1.weight'].size(1),
                hidden_size=checkpoint['model_state_dict']['fc1.weight'].size(0),
                latent_size=checkpoint['model_state_dict']['mu.weight'].size(0))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer



def save_output_images(images, epoch, output_dir='images'):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img = img.view(3, 128, 128).detach().cpu().numpy()
        img = (img * 255).astype('uint8')  
        img = img.transpose(1, 2, 0) 
        img_pil = Image.fromarray(img)
        img_pil.save(f"{output_dir}/epoch_{epoch}_img_{i}.png")
