import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv =
  def forward(self, img) -> ("D_content", "D_layout", "D_low_level"):
    assert(len(img.shape) == 3), "Expected shape [RGB, H, W]"
    assert(img.shape[0] == 3), "Expected shape [RGB, H, W]"
    raise NotImplementedError()

class Generator(nn.Module):
  def __init__(
    self,
    latent_size: int = 64,
  ):
    super().__init__()
    self.latent_size = latent_size
  def forward(self, latent) -> ["RGB", "H'", "W'"]:
    raise NotImplementedError()
  def generate_latent(self, device="cpu"):
    return torch.randn(self.latent_size, device=device, dtype=torch.float)
