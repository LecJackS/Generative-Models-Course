# %% [markdown]
"""
# Variational Autoencoder (VAE) – Practical Session

This notebook is the main practical entry point to VAEs for the course.

You will:

- Build and train a small VAE on MNIST (or Fashion-MNIST).
- Implement the **reparameterization trick**.
- Implement the **VAE loss** (reconstruction + KL).
- Visualize reconstructions, samples, and the latent space.
- Make connections to hierarchical VAEs and diffusion models.

The notebook is designed for a **4-hour session**:

- ~30–60 minutes theory mini-talk (can also be inferred from this notebook).
- ~2.5–3 hours of hands-on work.

> Important: cells with `# TODO` are meant to be completed by you.
> Try to work through them in order; they build on each other.
"""

# %% [markdown]
"""
## Section 0 – Recap: What is a VAE?

This section is a **self-contained refresher** on VAEs.  
If you missed the mini-lecture, read this carefully before you start coding.

---

### 0.1 The Problem: Generating High-Dimensional Data

We want a model that can **generate new images** that look like the ones in our dataset.  
Mathematically, we would like to model a complicated distribution over images, $p(x)$, where $x$ is a point in a very high-dimensional space (e.g. $28 \times 28 = 784$ dimensions for MNIST).

Directly modeling $p(x)$ is hard. Instead, we introduce a lower-dimensional **latent variable** $z$ which captures abstract factors of variation (e.g. style, stroke thickness, digit identity).

---

### 0.2 Generative Story of a VAE

The VAE assumes a simple **prior** over latent variables:

- Prior: $p(z) = \mathcal{N}(0, I)$

Then it defines a **decoder** (a neural network) that maps $z$ back to data space:

- Generative model (decoder): $p_\theta(x \mid z)$

Together, they define a joint distribution:

$$
p_\theta(x, z) = p(z) \, p_\theta(x \mid z).
$$

To **generate a new image**, we:

1. Sample $z \sim p(z)$ (e.g. a 2D or 10D Gaussian).
2. Pass $z$ through the decoder to get $p_\theta(x \mid z)$.
3. Sample or take the mean of that distribution to produce an image.

This gives a powerful, flexible generator, but we still face one problem:  
given an observed image $x$, we would like to know **which $z$ produced it**.

---

### 0.3 Inference Model: Encoder and Approximate Posterior

The exact posterior $p_\theta(z \mid x)$ is usually intractable.  
The VAE introduces an **encoder** network to approximate it:

- Encoder: $q_\phi(z \mid x)$

You can think of $q_\phi(z \mid x)$ as a **learned, data-dependent Gaussian** over $z$.  
For each image $x$, the encoder outputs:

- A mean vector $\mu_\phi(x)$
- A log-variance vector $\log \sigma_\phi^2(x)$

so that

$$
q_\phi(z \mid x) = \mathcal{N}\big(z; \mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x))\big).
$$

This encoder will play a central role in the implementation.

---

### 0.4 Training Objective: ELBO in Words

The VAE is trained by maximizing an **Evidence Lower Bound (ELBO)** on the log-likelihood $\log p_\theta(x)$:

$$
\log p_\theta(x) \ge
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
 - \mathrm{KL}\big(q_\phi(z \mid x) \,\Vert\, p(z)\big).
$$

This bound decomposes into two terms:

- **Reconstruction term**: $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$
  - Encourages the decoder to reconstruct $x$ well from $z$.
  - In practice, this becomes a pixel-wise loss (e.g. binary cross-entropy or MSE).
- **KL term**: $\mathrm{KL}\big(q_\phi(z \mid x) \,\Vert\, p(z)\big)$
  - Encourages the approximate posterior to stay close to the prior.
  - Keeps the latent space nicely shaped (e.g. close to a standard Gaussian).

You can think of the VAE as a **regularized autoencoder**:

- It wants **good reconstructions**, like a normal autoencoder.
- But it also wants a **smooth, regular latent space** where neighboring $z$ values decode to similar images.

We implement this with neural networks:

- **Encoder network** $x \to (\mu_\phi(x), \log \sigma^2_\phi(x))$ parameterizes $q_\phi(z \mid x)$.
- **Decoder network** $z \to \text{parameters of } p_\theta(x \mid z)$ parameterizes $p_\theta(x \mid z)$.

Later in the notebook, you will implement these networks and the corresponding loss terms.
"""

# %% [markdown]
"""
### 0.5 The Reparameterization Trick (Core Idea)

Our encoder defines a **diagonal Gaussian** approximate posterior:

$$
q_\phi(z \mid x) = \mathcal{N}\big(z; \mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x))\big).
$$

To evaluate the ELBO, we need to:

1. Sample $z \sim q_\phi(z \mid x)$
2. Pass $z$ to the decoder to compute $\log p_\theta(x \mid z)$
3. Backpropagate gradients through this whole process.

If we sample $z$ directly from the Gaussian, the sampling operation is **not a simple differentiable function of $\mu_\phi(x)$ and $\sigma_\phi(x)$**, which makes gradient-based training tricky.

The **reparameterization trick** rewrites the sampling step as a deterministic transformation of noise:

1. Sample $\varepsilon \sim \mathcal{N}(0, I)$ (independent of $x$ and $\phi$).
2. Compute

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon.
$$

Now:

- $\varepsilon$ is random **but fixed-distribution** noise.
- $z$ is a **differentiable function** of $(\mu_\phi(x), \sigma_\phi(x))$.

This means we can write code like:

```python
eps = torch.randn_like(mu)           # ε ~ N(0, I)
std = torch.exp(0.5 * log_var)       # σ = exp(0.5 * log σ²)
z = mu + std * eps                   # z = μ + σ ⊙ ε
```

and PyTorch will correctly propagate gradients back into the encoder parameters $\phi$.

You will implement this operation in the `reparameterize` function later in the notebook.
"""

# %% [markdown]
"""
### 0.6 Toy Example: Latent vs Data Space (Optional)

Before we go to real images, it is helpful to visualize the idea of a **simple latent space** mapping to a **more complex data space**.

In the next cell, we will:

- Create a toy 2D dataset (a mixture of a few Gaussians).
- Sample points from a standard 2D Gaussian (our latent prior).
- Plot both side by side.

The goal is to have a mental picture of:

- A **simple** distribution over $z$ (standard normal).
- A potentially **messy** distribution over $x$ (mixture of modes).

Important:

- In this toy cell we **do not yet connect** the two plots with a learned mapping.
- We simply place side by side:
  - a structured, multimodal **data distribution** $p_{\text{data}}(x)$ (left), and
  - a simple **latent prior** $p(z) = \mathcal{N}(0, I)$ (right),
  which is chosen by design in VAEs.
- Later, the VAE decoder will learn a function $f_\theta : z \mapsto x$ such that
  samples $z \sim \mathcal{N}(0, I)$ are mapped to points that follow the data distribution.

So the point of this example is **visual intuition**: we want a simple latent distribution that,
after passing through a flexible decoder, can approximate a complicated data distribution.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# Build a synthetic 2D "data space" distribution: a mixture of three Gaussians.
# This stands in for a complex, multimodal image distribution that the model
# should ultimately learn to approximate.
rng = np.random.default_rng(0)
n_per_cluster = 200
means = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])
cov = 0.05 * np.eye(2)
data_points = np.concatenate(
    [rng.multivariate_normal(m, cov, size=n_per_cluster) for m in means],
    axis=0,
)

# Independently, sample from a simple latent prior: a standard 2D normal N(0, I).
# This is the distribution we will use later for z in the VAE.
latent_points = rng.standard_normal(size=(3 * n_per_cluster, 2))

# Visualize both distributions side by side to emphasize:
# - left: a structured, multimodal data distribution p(x)
# - right: a simple, isotropic latent prior p(z)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].scatter(
    data_points[:, 0],
    data_points[:, 1],
    s=5,
    alpha=0.6,
    color="tab:blue",
)
axes[0].set_title("Toy data space x (mixture of clusters)")
axes[0].set_xlabel("x₁")
axes[0].set_ylabel("x₂")

axes[1].scatter(
    latent_points[:, 0],
    latent_points[:, 1],
    s=5,
    alpha=0.6,
    color="tab:orange",
)
axes[1].set_title("Latent space z ~ N(0, I)")
axes[1].set_xlabel("z₁")
axes[1].set_ylabel("z₂")

plt.tight_layout()
plt.show()

# %% [markdown]
"""
### Implementation Plan for This Notebook

We will proceed in the following steps:

1. Load and visualize the dataset (MNIST or Fashion-MNIST).
2. Implement **Encoder** and **Decoder** networks.
3. Implement the **reparameterization trick**.
4. Implement the **VAE loss** (reconstruction + KL).
5. Train the VAE and inspect reconstructions and samples.
6. Explore the latent space and perform interpolation.
7. (Optional) Explore β-VAE and different latent dimensions; connect to hierarchical VAEs and diffusion.
"""

# %% [markdown]
"""
## Section 1 – Setup and Dataset

In this section you will:

- Import dependencies.
- Set up device and configuration.
- Load the dataset and visualize a few samples.
"""

# %%
import math
import random
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms, utils as vutils
except ImportError as e:
    raise ImportError(
        "torchvision is required for this notebook. "
        "Please install it with `pip install torchvision`."
    ) from e


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_seed(42)

# %%
# Configuration (you can tweak these)
DATASET_NAME = "MNIST"  # or "FashionMNIST"
BATCH_SIZE = 128
LATENT_DIM = 2  # 2 is nice for visualization; try 10 or 32 later
HIDDEN_DIM = 512
NUM_EPOCHS = 5  # keep training short for class
LEARNING_RATE = 1e-3

image_size = 28 * 28  # for MNIST-like datasets

print(
    f"Config: dataset={DATASET_NAME}, batch_size={BATCH_SIZE}, "
    f"latent_dim={LATENT_DIM}, hidden_dim={HIDDEN_DIM}"
)

# %% [markdown]
"""
### Load Dataset

We will use MNIST (or Fashion-MNIST) from `torchvision.datasets`.

Images will be:

- Converted to tensors in $[0, 1]$.
- Flattened later when fed into the MLP-based encoder/decoder.
"""

# %%
def get_datasets(name: str = DATASET_NAME, root: str = "./data"):
    if name.upper() == "MNIST":
        dataset_cls = datasets.MNIST
    elif name.upper() == "FASHIONMNIST":
        dataset_cls = datasets.FashionMNIST
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1]
        ]
    )

    train_ds = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_ds = dataset_cls(root=root, train=False, download=True, transform=transform)
    return train_ds, test_ds


train_dataset, test_dataset = get_datasets(DATASET_NAME)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# %% [markdown]
"""
### Visualize a Batch

Run the cell below to see a grid of sample images.
"""

# %%
import matplotlib.pyplot as plt


def show_batch(images: torch.Tensor, nrow: int = 8) -> None:
    """Utility to visualize a batch of images."""

    grid = vutils.make_grid(images[: nrow * nrow], nrow=nrow, padding=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


examples, labels = next(iter(train_loader))
print(f"Batch shape: {examples.shape}, labels shape: {labels.shape}")
show_batch(examples)

# %% [markdown]
"""
## Section 2 – Encoder, Decoder, and VAE Wrapper

We now define the neural networks that parameterize our VAE:

- **Encoder**: maps input images $x$ to Gaussian parameters $(\mu(x), \log \sigma^2(x))$.
- **Decoder**: maps latent variables $z$ back to image logits.
- **VAE**: wraps encoder and decoder, and will later use the reparameterization trick.

We start with a simple MLP architecture:

- Flatten image $x \in \mathbb{R}^{28 \times 28}$ to $\mathbb{R}^{784}$.
- Use a hidden layer of size `HIDDEN_DIM`.
- Map to latent dimension `LATENT_DIM`.
"""

# %%
class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input x into mean and log-variance of q(z|x).

        Args:
            x: tensor of shape [batch_size, 1, 28, 28] or [batch_size, input_dim].

        Returns:
            mu: tensor of shape [batch_size, latent_dim]
            log_var: tensor of shape [batch_size, latent_dim]
        """
        # Ensure x is flattened
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # TODO: Implement the encoder forward pass.
        # 1. Apply a linear layer + nonlinearity.
        # 2. Map the hidden representation to mu and log_var.
        #
        # Hint:
        #   h = self.activation(self.fc1(x))
        #   mu = self.fc_mu(h)
        #   log_var = self.fc_logvar(h)
        #
        # Replace the lines below with your implementation.
        h = self.activation(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z into image logits.

        Args:
            z: tensor of shape [batch_size, latent_dim]

        Returns:
            logits: tensor of shape [batch_size, output_dim] (flattened image)
        """
        # TODO: Implement the decoder forward pass.
        # 1. Apply a linear layer + nonlinearity.
        # 2. Map to output_dim (image logits).
        #
        # Hint:
        #   h = self.activation(self.fc1(z))
        #   logits = self.fc_out(h)
        #
        # Replace the lines below with your implementation.
        h = self.activation(self.fc1(z))
        logits = self.fc_out(h)
        return logits


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: input images of shape [batch_size, 1, 28, 28] or [batch_size, input_dim]

        Returns:
            recon_logits: logits for reconstructed images [batch_size, input_dim]
            mu: mean of q(z|x) [batch_size, latent_dim]
            log_var: log-variance of q(z|x) [batch_size, latent_dim]
        """
        mu, log_var = self.encoder(x)
        # NOTE: The actual sampling of z via reparameterization will be
        # implemented in a separate function `reparameterize(mu, log_var)`.
        # For now, we call that function (with a TODO implementation).
        z = reparameterize(mu, log_var)
        recon_logits = self.decoder(z)
        return recon_logits, mu, log_var


# Instantiate the model (we will train it later)
model = VAE(input_dim=image_size, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
print(model)

# %% [markdown]
"""
## Section 3 – Reparameterization and VAE Loss

This is the core conceptual section of the notebook.

We will:

1. Implement the **reparameterization trick** to sample $z \sim q_\phi(z \mid x)$.
2. Implement the **VAE loss**, combining reconstruction and KL terms.
"""

# %%
def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick to sample z ~ N(mu, sigma^2) using epsilon ~ N(0, I).

    Args:
        mu: tensor of shape [batch_size, latent_dim]
        log_var: tensor of shape [batch_size, latent_dim]

    Returns:
        z: tensor of shape [batch_size, latent_dim]
    """
    # TODO: Implement the reparameterization trick.
    # Steps:
    # 1. Sample epsilon from N(0, I) with the same shape as mu.
    # 2. Compute sigma = exp(0.5 * log_var).
    # 3. Return z = mu + sigma * epsilon.
    #
    # Hint:
    #   eps = torch.randn_like(mu)
    #   std = torch.exp(0.5 * log_var)
    #   z = mu + std * eps
    #
    # Replace the lines below with your implementation.
    eps = torch.randn_like(mu)
    std = torch.exp(0.5 * log_var)
    z = mu + std * eps
    return z


def reconstruction_loss(recon_logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the reconstruction loss.

    We treat each pixel as a Bernoulli variable and use binary cross-entropy
    with logits. This assumes input x is in [0, 1].
    """
    # Flatten inputs if needed
    if x.dim() > 2:
        x = x.view(x.size(0), -1)

    # BCE with logits (stable implementation)
    bce = nn.functional.binary_cross_entropy_with_logits(
        recon_logits, x, reduction="none"
    )
    # Sum over features, mean over batch
    return bce.sum(dim=1).mean()


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Compute KL(q(z|x) || p(z)) where q is diagonal Gaussian and p is N(0, I).

    Closed-form for each dimension:
        KL = -0.5 * (1 + log_var - mu^2 - exp(log_var))

    We sum over dimensions and average over batch.
    """
    # TODO (optional): verify the KL formula on paper or from lecture notes.
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    kl = kl_per_dim.sum(dim=1).mean()
    return kl


def vae_loss(
    recon_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss:

        loss = recon_loss + beta * KL(q(z|x) || p(z))

    We return the total loss as well as its components so we can log them.
    """
    # TODO: Combine reconstruction and KL terms into a single scalar loss.
    # Steps:
    # 1. Compute reconstruction loss using `reconstruction_loss`.
    # 2. Compute KL divergence using `kl_divergence`.
    # 3. Combine:
    #       loss = recon + beta * kl
    #
    # Replace the lines below with your implementation.
    recon = reconstruction_loss(recon_logits, x)
    kl = kl_divergence(mu, log_var)
    loss = recon + beta * kl
    return loss, recon, kl


# %% [markdown]
"""
## Section 4 – Training Loop and Basic Evaluation

We now train the VAE using the loss defined above.

Training procedure:

1. For each mini-batch $x$:
   - Compute $(\text{recon\_logits}, \mu, \log \sigma^2) = \text{VAE}(x)$.
   - Compute loss = reconstruction + $\beta \cdot \text{KL}$.
   - Backpropagate and update parameters.
2. Periodically log the reconstruction and KL parts of the loss.
3. After training, visualize reconstructions and samples.
"""

# %%
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    beta: float = 1.0,
) -> None:
    model.train()
    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.to(device)

        optimizer.zero_grad()

        # TODO: Complete the training step for one batch.
        # Steps:
        # 1. Forward pass through the model to get recon_logits, mu, log_var.
        # 2. Compute loss, recon_loss, kl_loss using `vae_loss`.
        # 3. Backward pass and optimizer step.
        #
        # Hint:
        #   recon_logits, mu, log_var = model(x)
        #   loss, recon, kl = vae_loss(recon_logits, x, mu, log_var, beta=beta)
        #   loss.backward()
        #   optimizer.step()
        recon_logits, mu, log_var = model(x)
        loss, recon, kl = vae_loss(recon_logits, x, mu, log_var, beta=beta)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_recon += recon.item()
        running_kl += kl.item()

        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_recon = running_recon / (batch_idx + 1)
            avg_kl = running_kl / (batch_idx + 1)
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.2f} | Recon: {avg_recon:.2f} | KL: {avg_kl:.2f}"
            )


def train(
    num_epochs: int,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    beta: float = 1.0,
) -> None:
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch, model, dataloader, optimizer, beta=beta)


print("Ready to train. This should take a few minutes at most.")

# %%
# Uncomment this cell to start training once you have implemented all TODOs
# (reparameterize, vae_loss, and the training step).

# train(NUM_EPOCHS, model, train_loader, optimizer, beta=1.0)

# %% [markdown]
"""
### Evaluate Reconstructions

After training, we can pass some test images through the VAE and compare original and reconstructed images.
"""

# %%
def reconstruct_images(model: nn.Module, dataloader: DataLoader, num_images: int = 16):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x.to(device)
        recon_logits, mu, log_var = model(x)

        # Convert logits to probabilities in [0, 1] using sigmoid
        recon = torch.sigmoid(recon_logits)

        # Reshape to image shape
        recon = recon.view(-1, 1, 28, 28)

    # Show originals and reconstructions
    print("Original images:")
    show_batch(x.cpu(), nrow=int(math.sqrt(num_images)))
    print("Reconstructed images:")
    show_batch(recon.cpu(), nrow=int(math.sqrt(num_images)))


# Uncomment after training
# reconstruct_images(model, test_loader, num_images=16)

# %% [markdown]
"""
### Generate New Samples from the Prior

To generate new images, we:

1. Sample $z \sim p(z) = \mathcal{N}(0, I)$.
2. Decode $z$ via the decoder.
"""

# %%
def sample_from_prior(model: nn.Module, num_samples: int = 16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM, device=device)
        logits = model.decoder(z)
        samples = torch.sigmoid(logits)
        samples = samples.view(-1, 1, 28, 28)
    show_batch(samples.cpu(), nrow=int(math.sqrt(num_samples)))


# Uncomment after training
# sample_from_prior(model, num_samples=16)

# %% [markdown]
"""
## Section 5 – Latent Space Exploration and Interpolation

In low latent dimensions (e.g. 2), we can:

- Plot the latent representations of test images and color them by class.
- Interpolate between two points in latent space and decode the path.
"""

# %%
def encode_dataset(model: nn.Module, dataloader: DataLoader):
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            mu, log_var = model.encoder(x)
            z = mu  # use mean as the representation
            zs.append(z.cpu())
            ys.append(y)
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0)


def plot_latent_space(z: torch.Tensor, y: torch.Tensor, num_points: int = 5000):
    """Plot a 2D latent space with points colored by class label."""
    if z.size(1) != 2:
        print("Latent dimension is not 2; skipping 2D plot.")
        return

    z = z[:num_points]
    y = y[:num_points]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("Latent space (mean of q(z|x))")
    plt.show()


# Uncomment after training if LATENT_DIM == 2
# z_all, y_all = encode_dataset(model, test_loader)
# plot_latent_space(z_all, y_all)

# %%
def interpolate(z1: torch.Tensor, z2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """
    Linearly interpolate between two latent points z1 and z2.

    Args:
        z1: [latent_dim]
        z2: [latent_dim]
        num_steps: number of interpolation steps (including endpoints)

    Returns:
        zs: [num_steps, latent_dim] tensor of interpolated points
    """
    # TODO: Implement linear interpolation between z1 and z2.
    # Steps:
    # 1. Create a tensor t of shape [num_steps] going from 0 to 1.
    # 2. Compute zs = (1 - t) * z1 + t * z2 (broadcasting over latent dim).
    #
    # Hint:
    #   t = torch.linspace(0, 1, steps=num_steps, device=z1.device).unsqueeze(1)
    #   zs = (1 - t) * z1.unsqueeze(0) + t * z2.unsqueeze(0)
    #
    # Replace the lines below with your implementation.
    t = torch.linspace(0.0, 1.0, steps=num_steps, device=z1.device).unsqueeze(1)
    zs = (1 - t) * z1.unsqueeze(0) + t * z2.unsqueeze(0)
    return zs


def visualize_interpolation(model: nn.Module, dataloader: DataLoader, num_steps: int = 10):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(dataloader))
        x = x.to(device)
        mu, log_var = model.encoder(x)
        z = mu

        # Pick two random examples
        z1 = z[0]
        z2 = z[1]

        zs = interpolate(z1, z2, num_steps=num_steps)
        logits = model.decoder(zs)
        samples = torch.sigmoid(logits).view(-1, 1, 28, 28)

    show_batch(samples.cpu(), nrow=num_steps)


# Uncomment after training
# visualize_interpolation(model, test_loader, num_steps=10)

# %% [markdown]
"""
### Reflection Questions (Answer in Markdown)

1. When you plot the latent space (for $d_z = 2$), do you see clusters corresponding to different digits/classes?
2. How smooth are the interpolations in image space when you interpolate linearly in latent space?
3. If you increase the latent dimension (e.g. to 10 or 32), how do reconstructions and samples change?
4. What happens to the model behavior if you change the β parameter in the VAE loss (e.g. β = 0.1 vs β = 4)?
"""

# %% [markdown]
"""
## Section 6 – Optional Extensions and Connections

### 6.1 β-VAE Experiment (Optional)

Try the following:

1. Change the `beta` parameter in `train` and `vae_loss`.
2. Train models with different β (e.g. 0.5, 1.0, 4.0).
3. Compare:
   - Reconstruction quality.
   - Latent space structure.

Heuristics:

- Smaller β (< 1): better reconstructions but less regular latent space.
- Larger β (> 1): more regular latent space but possibly worse reconstructions.
"""

# %% [markdown]
"""
### 6.2 Towards Hierarchical VAEs

In this notebook, we used a **single latent layer** $z$.

Hierarchical VAEs introduce multiple latent layers, such as:

$$
z_2 \sim p(z_2), \quad
z_1 \sim p(z_1 \mid z_2), \quad
x \sim p(x \mid z_1).
$$

Key points:

- The same ideas reappear:
  - Encoder networks approximating posteriors at each latent level.
  - Reparameterization trick to sample from each approximate posterior.
  - ELBO with reconstruction terms and KL terms at each layer.
- The latent space becomes richer and can capture structure at different scales.

In the next session, we will build on this notebook to implement a simple hierarchical VAE.
"""

# %% [markdown]
"""
### 6.3 Connecting to Diffusion Models

VAEs and diffusion models are different generative modeling paradigms, but they share some important themes:

- **Noise and uncertainty**:
  - VAEs: latent variable $z$ with Gaussian noise.
  - Diffusion: a sequence of noisy variables $x_t$ obtained by gradually adding noise.
- **Learning to reverse a stochastic process**:
  - VAEs: decode from $z$ back to data space.
  - Diffusion: denoise from $x_T$ back to $x_0$.
- **Likelihood-based training**:
  - VAEs: train via the ELBO.
  - Diffusion: train via objectives related to log-likelihood and score matching.

The intuition you build here—about sampling from a prior, decoding to data space, and balancing reconstruction vs. regularization—will be important when you study diffusion models later in the course.
"""
