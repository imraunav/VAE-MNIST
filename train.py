import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import os

from model import VariationalAutoEncoder


def inference(digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"samples/generated_{digit}_ex{example}.png")


os.makedirs("./samples", exist_ok=True)

# Configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 28 * 28
HIDDEN_DIM = 200
LATENT_DIM = 20
NUM_EPOCHS = 25
BATCH_SIZE = 64

LEARNING_RATE = 3e-4  # Karpathy constant

# Dataset and model prep
dataset = datasets.MNIST(
    "./dataset", train=True, transform=transforms.ToTensor(), download=True
)
dataloder = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
crit = nn.BCELoss(reduction="sum")
alpha = 1
beta = 1

model.train()
for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}")
    loop = tqdm(dataloder)
    for imgs, labels in loop:
        # forward
        imgs = imgs.to(DEVICE).view(
            imgs.size(0), INPUT_DIM
        )  # flatten samples in a batch
        recon, mu, sigma = model(imgs)

        # loss
        recon_loss = crit(recon, imgs)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # bacward
        loss = alpha * recon_loss + beta * kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
torch.save(model.state_dict(), "vae-mnist.pt")

model.load_state_dict(torch.load("vae-mnist.pt"))
model.eval()

for idx in range(10):
    inference(idx, num_examples=5)