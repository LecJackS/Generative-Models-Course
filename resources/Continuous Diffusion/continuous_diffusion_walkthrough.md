# Continuous Diffusion: Step-by-Step Guide (with Q&A)

Use this as a teaching-ready walkthrough for the two “Continuous Diffusion” notebooks (`W2D4_Tutorial2.ipynb` and `W2D4_Tutorial3.ipynb`). It keeps the original structure, fills in the missing pieces, and adds checkpoints with hidden answers for students. Code snippets are PyTorch/NumPy and match the notebooks; you can paste them back if cells get reset.

---

## 0) Setup and environment

- GPU helps but CPU also works for the toy parts. MNIST training will be slower on CPU.
- Fix seeds for reproducibility:

```python
import random, torch, numpy as np

def set_seed(seed=2021, seed_torch=True):
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
```

- Device helper:

```python
def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("TIP: Switch runtime to GPU if available.")
    return device
```

---

## 1) Forward diffusion (Variance Exploding SDE intuition)

We use the VE SDE:
\[
dx = g(t)\,dW_t,\quad g(t) = \lambda^t
\]
Noise accumulation (variance):
\[
\sigma_t^2 = \frac{\lambda^{2t} - 1}{2 \ln \lambda},\qquad
\sigma_t = \sqrt{\frac{\lambda^{2t} - 1}{2 \ln \lambda}}.
\]

```python
import numpy as np

def sigma_t_square(t, Lambda):
    return (Lambda ** (2 * t) - 1) / (2 * np.log(Lambda))

def sigma_t(t, Lambda):
    return np.sqrt(sigma_t_square(t, Lambda))
```

Simple 1D forward simulation (bimodal start, then diffusion):

```python
import matplotlib.pyplot as plt

def diffusion_1d_forward(Lambda=20, timesteps=100, sampleN=200):
    t = np.linspace(0, 1, timesteps)
    dt = t[1] - t[0]
    dw = np.random.normal(0, np.sqrt(dt), size=(timesteps, sampleN))
    x0 = np.concatenate([
        np.random.normal(-5, 1, size=sampleN // 2),
        np.random.normal(5, 1, size=sampleN - sampleN // 2)
    ])
    x = np.cumsum((Lambda ** t[:, None]) * dw, axis=0) + x0[None, :]
    plt.plot(t, x[:, : sampleN // 2], color="r", alpha=0.1)
    plt.plot(t, x[:, sampleN // 2 :], color="b", alpha=0.1)
    plt.xlabel("time"); plt.ylabel("x"); plt.title(f"Forward diffusion, λ={Lambda}")
    plt.show()
```

**Checkpoint (answers hidden)**  
1) Why VE SDE (growing variance) instead of a constant-noise SDE?  
<details><summary>Answer</summary>Large variance at late times makes the reverse process well-conditioned and easier to discretize; VE connects to score matching with simple Gaussian perturbations.</details>

2) What happens to the data distribution as \(t\to 1\)?  
<details><summary>Answer</summary>It approaches a wide Gaussian \(N(0,\sigma_T^2 I)\), losing fine structure but keeping a tractable form for initialization.</details>

---

## 2) Score function on a Gaussian mixture (intuition builder)

- The notebook defines a small `GaussianMixture` helper to sample and evaluate log-densities.
- The **score** is \(\nabla_x \log p(x)\); for mixtures it points toward nearby modes and away from low-density gaps.

Example: compute and plot the score field for a 2D mixture:

```python
from scipy.stats import multivariate_normal
import numpy as np

class GaussianMixture:
    def __init__(self, mus, covs, weights):
        self.mus = mus
        self.covs = covs
        self.precs = [np.linalg.inv(c) for c in covs]
        self.weights = np.array(weights) / np.sum(weights)
        self.RVs = [multivariate_normal(m, c) for m, c in zip(mus, covs)]
        self.dim = len(mus[0])

    def logpdf(self, x):
        return np.log(np.sum([w * rv.pdf(x) for w, rv in zip(self.weights, self.RVs)], axis=0))

    def score(self, x):
        # gradient of log mixture density
        num = 0
        denom = 0
        for w, m, prec, rv in zip(self.weights, self.mus, self.precs, self.RVs):
            p = w * rv.pdf(x)
            num += (x - m) @ prec.T * p
            denom += p
        return -num / (denom[:, None] + 1e-8)
```

**Checkpoint**  
How does the score behave near (1) a mode, (2) a low-density gap?  
<details><summary>Answer</summary>(1) Vectors shrink toward zero near a mode; (2) vectors point toward the closest mode, giving large magnitude in valleys.</details>

---

## 3) Reverse diffusion with a known score (GMM)

Reverse SDE (VE):
\[
d x_t = \big[g(t)^2 \nabla_x \log p_t(x_t)\big] dt + g(t)\, d\bar{W}_t.
\]

Discretized sampler for the GMM score:

```python
def reverse_diffusion_SDE_sampling_gmm(gmm, sampN=500, Lambda=5, nsteps=500):
    sigmaT2 = sigma_t_square(1, Lambda)
    x = np.sqrt(sigmaT2) * np.random.randn(sampN, 2)
    traj = np.zeros((*x.shape, nsteps))
    traj[:, :, 0] = x
    dt = 1 / nsteps
    for i in range(1, nsteps):
        t = 1 - i * dt
        eps = np.random.randn(*x.shape)
        g = Lambda ** t
        score = diffuse_gmm(gmm, t, Lambda).score(traj[:, :, i - 1])
        traj[:, :, i] = traj[:, :, i - 1] + eps * g * np.sqrt(dt) + score * dt * g**2
    return traj
```

**Checkpoint**  
Why do we still add noise \(g(t)\sqrt{dt}\,z\) in reverse-time sampling?  
<details><summary>Answer</summary>The reverse SDE is still stochastic; removing the noise term produces an ODE variant (probability flow ODE) but changes the sampler.</details>

---

## 4) Denoising Score Matching (DSM) objective

Given data \(x\), sample \(t\sim U[\varepsilon,1]\), add Gaussian noise with std \(\sigma_t\), and train:
\[
\mathbb{E}\|\sigma_t\, s_\theta(x+\sigma_t z, t) + z\|^2
\]

```python
import torch

def loss_fn(model, x, sigma_t_fun, eps=1e-5):
    batch = x.shape[0]
    t = torch.rand(batch, device=x.device) * (1 - eps) + eps
    z = torch.randn_like(x)
    std = sigma_t_fun(t)
    perturbed_x = x + z * std[:, None]
    score = model(perturbed_x, t)
    loss = torch.mean(torch.sum((score * std[:, None] + z) ** 2, dim=1))
    return loss
```

**Checkpoint**  
Why multiply the predicted score by \(\sigma_t\) in the loss?  
<details><summary>Answer</summary>The objective comes from denoising score matching: the optimal score for a Gaussian-perturbed point is \(-z/\sigma_t\); scaling aligns the predicted and true scores.</details>

---

## 5) Toy training: time-dependent MLP on 2D GMM

- Time embedding: Gaussian Fourier features (`GaussianFourierProjection`).
- Model: small MLP taking concatenated `x` and time features.
- Training loop sketch:

```python
score_model = ScoreModel_Time(sigma=25.0)  # defined in notebook
optim = torch.optim.Adam(score_model.parameters(), lr=5e-3)
sigma_t_f = lambda t: torch.sqrt(torch.tensor(sigma_t_square(t, 25.0)))

for step in range(500):
    loss = loss_fn(score_model, X_train_samp, sigma_t_f, eps=0.01)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

Then plug the trained model into `reverse_diffusion_SDE_sampling` (use `exact=False`) to visualize sample trajectories converging to the mixture modes.

---

## 6) Image diffusion on MNIST (UNet score model)

Same VE SDE, now on images.

```python
import torch

def marginal_prob_std(t, Lambda, device="cpu"):
    t = t.to(device)
    lam = torch.tensor(Lambda, device=device, dtype=t.dtype)
    return torch.sqrt((lam ** (2 * t) - 1) / (2 * torch.log(lam)))

def diffusion_coeff(t, Lambda, device="cpu"):
    t = t.to(device)
    lam = torch.tensor(Lambda, device=device, dtype=t.dtype)
    return lam ** t
```

DSM loss for images:

```python
def loss_fn(model, x, marginal_prob_std, eps=1e-3, device="cpu"):
    t = torch.rand(x.shape[0], device=device) * (1.0 - eps) + eps
    std = marginal_prob_std(t).to(device)
    z = torch.randn_like(x)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1,2,3)))
    return loss
```

Network: time-embedded UNet (`UNet` in the notebook) with Gaussian Fourier time embeddings and standard down/up blocks. Train with MNIST, Adam, cosine LR schedule (see `Training the model` cell).

**Checkpoint**  
What happens if \(\Lambda\) is too small or too large?  
<details><summary>Answer</summary>Too small: insufficient noise, reverse process ill-posed; too large: high noise makes score learning harder and may demand more steps.</details>

---

## 7) Sampling via Euler–Maruyama (reverse SDE)

From the trained score model:

```python
def Euler_Maruyama_sampler(score_model, marginal_prob_std, diffusion_coeff,
                           batch_size=64, x_shape=(1,28,28),
                           num_steps=500, device="cuda", eps=1e-3, y=None):
    t = torch.ones(batch_size, device=device)
    x = torch.randn(batch_size, *x_shape, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    dt = time_steps[0] - time_steps[1]
    with torch.no_grad():
        for ts in time_steps:
            bt = torch.ones(batch_size, device=device) * ts
            g = diffusion_coeff(bt)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, bt, y=y) * dt
            x = mean_x + torch.sqrt(dt) * g[:, None, None, None] * torch.randn_like(x)
    return mean_x  # last mean (no noise)
```

Visualize a grid:

```python
samples = Euler_Maruyama_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn,
                                 batch_size=64, num_steps=250, device=DEVICE)
samples = samples.clamp(0, 1)
grid = torchvision.utils.make_grid(samples, nrow=8)
plt.imshow(grid.permute(1,2,0).cpu()); plt.axis("off"); plt.show()
```

---

## 8) Conditional diffusion on digits

- Conditional UNet: same as above, plus an embedding for class labels (`UNet_Conditional`).
- Conditional loss:

```python
def loss_fn_cond(model, x, y, marginal_prob_std, eps=1e-3):
    t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, t, y=y)
    return torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1,2,3)))
```

Sampling with a fixed label:

```python
digit = 4
samples = Euler_Maruyama_sampler(score_model_cond, marginal_prob_std_fn, diffusion_coeff_fn,
                                 batch_size=64, num_steps=250, device=DEVICE,
                                 y=torch.full((64,), digit, device=DEVICE, dtype=torch.long))
```

**Checkpoint**  
How does classifier-free guidance relate to this conditional setup?  
<details><summary>Answer</summary>It blends conditional and unconditional scores to steer samples; this notebook uses pure conditional embeddings but the same sampler can be extended with guidance.</details>

---

## 9) Stable Diffusion demo and ethics

- Notebook downloads `stabilityai/stable-diffusion-2-1` via `diffusers`, swaps in DPM-Solver++ scheduler, and runs text-to-image.
- Use `recursive_print` helper to inspect UNet/text encoder depth-limited.
- Include a short discussion on copyright/ethical considerations.

---

## 10) Suggested class flow

1) Concept/intuition: forward VE SDE, score fields on GMM.  
2) Hands-on: implement DSM loss, train tiny score net on GMM, sample with reverse SDE.  
3) Scale up: derive `marginal_prob_std`, implement image DSM loss, train UNet on MNIST.  
4) Sample with Euler–Maruyama; discuss artifacts vs. step size and \(\Lambda\).  
5) Extend: conditional sampling; peek at Stable Diffusion and ethics.

Encourage students to answer checkpoint questions before opening the `<details>` answers. Keep runtimes short in class (small epochs, fewer sampling steps) and point to the same code paths for deeper experiments. 
