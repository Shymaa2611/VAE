import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import save_checkpoint,save_output_images,loss_criterion
from dataset import UnsupervisedImageDataset
from model import VAE



def train(dataloader, model, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in dataloader:
            inputs = inputs.view(inputs.size(0), -1) 
            optimizer.zero_grad()
            recon_x, logvar, mu, z = model(inputs)
            loss = loss_criterion(recon_x, inputs, logvar, mu)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}')
        #if epoch % 1 == 0:  
        if epoch==200:
            with torch.no_grad():
                sample = next(iter(dataloader))[0].view(-1, 128 * 128 * 3)
                reconstructed, _, _, _ = model(sample)
                save_output_images(reconstructed, epoch)


def main():
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor()  
    ])
    dataset = UnsupervisedImageDataset(root_dir='data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = 128 * 128 * 3 
    hidden_size = 512
    latent_size = 30
    model = VAE(input_size, hidden_size, latent_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(dataloader, model, optimizer, epochs=200)
    save_checkpoint(model, optimizer, 'checkpoint/checkpoint.pt')


if __name__ == "__main__":
    main()
