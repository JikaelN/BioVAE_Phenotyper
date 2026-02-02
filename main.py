import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

import numpy as np
import os
from tqdm import tqdm

#local modules
from src.model_pytorch import BioVAE
from src.utils_jax import jax_kl_divergence

# --- Configurations ---
CONFIG = {
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 1e-3,
    "EPOCHS": 50,
    "DATA_FLAG": "bloodmnist", # Blood cells
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SAVE_DIR": "checkpoints"
}


def get_loaders(data_flag, batch_size):
    """
    Standardized data loading pipeline using MedMNIST datasets.
    return train and test loaders.
    """
    
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    #Preprocessing: Convert to Tensor and Normalize
    import torchvision.transforms as transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def loss_function(recon_x, x, mu, logvar):
    """
    Docstring for loss_function
    
    :param recon_x: Description
    :param x: Description
    :param mu: Description
    :param logvar: Description
    """
    
    # BCE = Binary Cross Entropy Loss
    # Using MSE here because input are normalized to [-1, 1] or [0, 1]
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KLD = KullbackLeibler Divergence
    # -0.5 * torch.sum(1 + logvar- mu.pow(2) - logvar.exp())
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD, BCE, KLD

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    
    # Use tqdm for progress bar
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['EPOCHS']} [Training]")
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(CONFIG['DEVICE'])
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute loss
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        
        # Backward pass and optimization
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        # Update tqdm loop description
        loop.set_postfix(loss=loss.item() / len(data), bce=bce.item() / len(data), kld=kld.item() / len(data))
    
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")
    
    
def validate_with_jax_check(model, test_loader):
    """
    Validation loop that also uses JAX to verify calculations.
    """
    
    model.eval()
    test_loss = 0
    jax_consistency_checks = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(CONFIG['DEVICE'])
            recon_batch, mu, logvar = model(data)
            
            
            # 1. PyTorch loss calculation
            loss, bce, kld_torch = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            
            # 2. JAX loss calculation for KLD
            # We take tensors from Pytorch, convert to numpy and feed it to JAX
            mu_np = mu.cpu().numpy()
            logvar_np = logvar.cpu().numpy()
            
            # Call our JAX function (JIT compiled)
            kld_jax = jax_kl_divergence(mu_np, logvar_np)
            
            # Check difference
            diff = np.abs(kld_torch.cpu().numpy() - kld_jax)
            jax_consistency_checks.append(diff)
            
        avg_loss = test_loss / len(test_loader.dataset)
        avg_jax_diff = np.mean(jax_consistency_checks)
        
        print(f"====> Test set loss: {avg_loss:.4f}")
        print(f"====> Average JAX-PyTorch KLD difference: {avg_jax_diff:.6f}")
        
        
def main():
    # Ensure reproducibility
    torch.manual_seed(42)
    
    print(f"Starting BioVEA on device: {CONFIG['DEVICE']}")
    
    # 1. Load Data
    train_loader, test_loader = get_loaders(CONFIG['DATA_FLAG'], CONFIG['BATCH_SIZE'])
    
    # 2. Setup Model
    model = BioVAE().to(CONFIG['DEVICE'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    
    # 3. Training Loop
    os.makedirs(CONFIG['SAVE_DIR'], exist_ok=True)
    
    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        train(model, train_loader, optimizer, epoch)
        validate_with_jax_check(model, test_loader)
        
    save_path = os.path.join(CONFIG["SAVE_DIR"], "biovae_bloodmnist.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
            