# src/utils_jax.py
import jax.numpy as jnp
from jax import jit

@jit
def jax_kl_divergence(mu, logvar): # Loss function metrics
    """Compute the KL divergence between the learned latent distribution and a standard normal distribution."""
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar))