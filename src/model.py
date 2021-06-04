import torch
import torch.nn as nn
import torchvision

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
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
    self.resnet18 = torchvision.models.resnet18(pretrained=True)
  def forward(self, latent) -> ["RGB", "H'", "W'"]:
    raise NotImplementedError()

    x = latent
    assert(x.shape[0] == 3)
    return x
  def generate_latent(self):
    return torch.randn(self.latent_size, device=self.device, dtype=torch.float)
