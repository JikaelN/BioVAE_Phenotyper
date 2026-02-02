import torch
from src.model_pytorch import BioVAE

def test_model_output_shape():
    """Ensure the model outputs images of the same size as input."""
    model = BioVAE()
    input_tensor = torch.randn(4, 3, 28, 28)  # Batch of 4 images, 3 channels, 28x28 size
    dummy_input = torch.randn(1, 3, 28, 28) # Batch size 1, 3 channels, 28x28
    recon, mu, logvar = model(dummy_input)
    
    assert recon.shape == dummy_input.shape, f"Expected output shape {dummy_input.shape}, but got {recon.shape}"
    assert mu.shape == (1, 20), f"Expected mu shape (1, 20), but got {mu.shape}"