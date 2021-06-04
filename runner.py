import argparse
import torch
import torch.optim as optim
import torchvision
from tqdm import trange
from itertools import chain

from src.model import ( Generator, Discriminator )

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("-i", "--input", help="Input image to train on", required=True, type=str)
  a.add_argument("--epochs", help="Number of training epochs", type=int, default=10_000)
  a.add_argument(
    "-lr", "--learning-rate", help="Learning rate for model", type=float, default=5e-3
  )
  return a.parse_args()

device = "cpu"
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

def load_image(args):
  return torchvision.io.read_image(args.input)/255

def model():
  disc = Discriminator().to(device)
  gen = Generator().to(device)
  return gen, disc

def train(img, gen, disc, opt, args):
  t = trange(args.epochs)
  for i in t:
    opt.zero_grad()
    # generator step
    img = gen(gen.generate_latent())
    img
    # discriminator step
    d_content, d_layout, d_low_level = disc(img)
    adversarial = d_content + d_layout + 2 * d_low_level
    adversarial.backward()

    opt.step()

def test(img, gen, disc, args):
  ...

def run():
  args = arguments()

  # load image
  img = load_image(args)
  # create GAN model
  gen, disc = model()

  opt = optim.Adam(chain(gen.parameters(), disc.parameters()), lr=args.learning_rate)
  # train model
  train(img, gen, disc, opt, args)
  # test model
  test(img, gen, disc, args)

if __name__ == "__main__": run()
