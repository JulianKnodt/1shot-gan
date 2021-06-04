import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from tqdm import trange

from src.model import ( Generator, Discriminator )

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("-i", "--input", help="Input image to train on", required=True, type=str)
  a.add_argument("--epochs", help="Number of training epochs", type=int, default=10_000)
  a.add_argument(
    "-lr", "--learning-rate", help="Learning rate for model", type=float, default=5e-3
  )
  a.add_argument("--batch-size", help="Training batch size", type=int, default=6)
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

def train(img, gen, disc, gen_opt, disc_opt, args):
  t = trange(args.epochs)
  for i in t:
    # train discriminator
    disc_opt.zero_grad()
    real_pred = disc(img)
    real_label = torch.ones(batch_size, device=device)
    real_loss = F.binary_cross_entropy_with_logits(real_pred, real_label)
    real_loss.backward()

    fake = gen(gen.generate_latent())
    fake_pred = disc(fake.detach()) # TODO need to clone after detach?
    fake_label = torch.zeros(batch_size, device=device)
    fake_loss = F.binary_cross_entropy_with_logits(fake_pred, fake_label)
    fake_loss.backward()
    disc_opt.step()
    # train generator
    gen_opt.zero_grad()
    pred = disc(fake)
    mimic_loss = F.binary_cross_entropy_with_logits(pred, real_label)
    mimic_loss.backward()
    gen_opt.step()
    t.set_postfix(
      mimic=f"{ditto_loss.item():.03f}",
      real=f"{real_loss.item():.03f}",
      fake=f"{fake_loss.item():.03f}",
      refresh=False,
    )
    if i % args.valid_freq == 0:
      fake_int8 = (fake * 255).byte()
      torchvision.io.write_jpeg(fake_int8, f"outputs/train_{i:05}.jpg")

def test(img, gen, disc, args):
  ...

def run():
  args = arguments()

  # load image
  img = load_image(args)
  # create GAN model
  gen, disc = model()

  gen_opt = optim.Adam(gen.parameters(), lr=args.learning_rate)
  disc_opt = optim.Adam(disc.parameters(), lr=args.learning_rate)
  # train model
  train(img, gen, disc, gen_opt, disc_opt, args)
  # test model
  test(img, gen, disc, args)

if __name__ == "__main__": run()
