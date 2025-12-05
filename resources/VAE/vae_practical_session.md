# VAE Practical Session Plan (4h)

This document designs the **practical VAE session** for the course.  
It assumes that separate theory lectures exist but may be asynchronous; the session must therefore be **self-contained** for any student with basic deep-learning and probability background.

Proposed main notebook filename: `01_VAE_practical.ipynb`.

---

## 1. Session Overview

Duration: ~4 hours total.

High-level structure:
- 0:00–0:10 – Setup and motivation (in class, slides or notebook markdown).
- 0:10–0:45 – VAE theory mini-talk (board/slides, can be mirrored in notebook markdown).
- 0:45–3:30 – Hands-on notebook work:
  - Build and train a small VAE on MNIST/Fashion-MNIST.
  - Analyze reconstructions, samples, and latent space.
  - Optional small extensions (β-VAE, changing latent dimension).
- 3:30–4:00 – Wrap-up:
  - Group discussion of results.
  - Short reflection questions.
  - Pointers to hierarchical VAEs and diffusion models.

Primary learning goals:
- Understand the **computational graph** of a VAE (encoder → latent → decoder).
- Implement the **reparameterization trick** in code.
- Implement the **VAE loss** (reconstruction + KL) and train a model.
- Interpret **latent space structure** via visualization and interpolation.
- See how this framework naturally leads to **hierarchical VAEs** and **diffusion models** in later sessions.

Assumed prerequisites:
- Basic familiarity with:
  - Neural networks and backprop.
  - Maximum likelihood / log-likelihood.
  - Gaussians, expectations, basic KL divergence notion (formal derivation can be reviewed in theory lectures).
- Experience with PyTorch or JAX for simple models.

---

## 2. 30–60 Minute Theory Mini-Talk (Outline)

This talk is given right before the hands-on part and should also be summarized in the notebook’s initial markdown cells so that students who miss the talk can still follow.

### 2.1 Motivation: Generative Latent Variable Models (5–10 min)
- Problem setting:
  - We want to learn a model that can **generate new data** similar to the training distribution.
  - Direct modeling of `p(x)` is hard for high-dimensional data (e.g. images).
- Latent-variable idea:
  - Introduce a latent variable `z` with simple prior `p(z)` (e.g. standard normal).
  - Define a generative process: `z ~ p(z)`, `x ~ p_θ(x | z)`.
  - This gives a rich family of distributions over `x`.
- Challenge:
  - Maximum likelihood involves `log p_θ(x) = log ∫ p_θ(x, z) dz`, typically intractable.

### 2.2 Variational Inference and ELBO (10–15 min)
- Approximate posterior:
  - Introduce `q_ϕ(z | x)` as a tractable approximation to the true posterior `p_θ(z | x)`.
  - This is the **encoder** network in the VAE.
- Derivation sketch of ELBO (high level, no need to dwell on details):
  - Start from `log p_θ(x)` and add/subtract `q_ϕ(z|x)`.
  - Arrive at:
    - `log p_θ(x) ≥ E_{q_ϕ(z|x)} [log p_θ(x | z)] - KL(q_ϕ(z|x) || p(z))`.
  - Interpret:
    - First term ≈ reconstruction quality.
    - Second term = regularizer enforcing approximate posterior close to prior.
- Key takeaway:
  - VAE training ≈ **maximize reconstruction quality while regularizing the latent space** via KL.

### 2.3 VAE Architecture and Reparameterization Trick (10–15 min)
- Architecture:
  - **Encoder**: `x → (μ_ϕ(x), log σ²_ϕ(x))`.
  - **Latent sampling**: `z = μ + σ ⊙ ε`, `ε ~ N(0, I)`.
  - **Decoder**: `z → p_θ(x | z)` (e.g. Bernoulli for normalized images or Gaussian).
- Reparameterization trick:
  - We need gradients w.r.t. ϕ through a sample from `q_ϕ(z|x)`.
  - Instead of sampling `z` directly from `N(μ, σ²)`, sample `ε ~ N(0, I)` and set:
    - `z = μ + σ ⊙ ε`.
  - This makes `z` a differentiable function of μ, σ, and ε; backprop can flow.
- Practical choices:
  - Prior: standard normal `p(z) = N(0, I)`.
  - Posterior: diagonal Gaussian `q_ϕ(z|x) = N(μ_ϕ(x), diag(σ²_ϕ(x)))`.
  - Output distribution and reconstruction loss (BCE vs MSE).

### 2.4 Practical Tips and Common Issues (5–10 min)
- KL term:
  - Analytic KL for two diagonal Gaussians → simple closed-form expression.
  - Can weight the KL term by β (β-VAE) to explore tradeoffs between reconstruction and regularization.
- Training tricks:
  - Start with small latent dimension (e.g. 2 or 10) for easy visualization.
  - Use simple architectures first; avoid over-parameterizing.
  - Monitor reconstruction loss and KL separately.
- Roadmap to later topics:
  - **Hierarchical VAEs**: add more latent layers (z₂ → z₁ → x).
  - **Diffusion models**: different generative paradigm, but still controlled by noise and probabilistic structure; they also require understanding latent/noise representations and likelihood/ELBO-like objectives.

---

## 3. Practical Notebook Design: `01_VAE_practical.ipynb`

The notebook must be **doable in ~2.5–3 hours** for most students, with:
- Minimal boilerplate to write.
- A few conceptually important coding tasks they must fill in.
- Optional extensions for faster/stronger students.

The tasks below are heavily inspired by common VAE tutorials (e.g. CS236, Agustinus Kristiadi’s blog, and VAE tutorial notebooks), but are recomposed and reworded for this specific course.

### High-Level Flow

Sections:
1. Section 0 – Self-contained VAE recap and setup.
2. Section 1 – Dataset and baseline visualization.
3. Section 2 – Model definition (Encoder, Decoder, VAE wrapper).
4. Section 3 – Reparameterization and VAE loss.
5. Section 4 – Training loop and basic evaluation.
6. Section 5 – Latent space exploration and interpolation.
7. Section 6 – Optional extensions and hooks to future topics.

Total student-coded lines (rough target): 80–150 lines, broken into small, well-scoped TODOs.

---

### 3.1 Section 0 – Self-Contained VAE Recap and Setup

Type: markdown + minimal code.

Goals:
- Recap core ideas for students who missed or forgot the mini-talk.
- Clarify notation used in the notebook.
- Set expectations about what they will implement.

Contents:
- Markdown:
  - One or two diagrams (included as images) showing the VAE computational graph:
    - Encoder → latent Gaussian → reparameterization → decoder.
  - Restate the ELBO at a conceptual level:
    - `ELBO(x) = E_q[log p_θ(x | z)] - KL(q_ϕ(z|x) || p(z))`.
  - Explain the roles of:
    - `p(z)` (prior).
    - `q_ϕ(z|x)` (encoder/posterior).
    - `p_θ(x|z)` (decoder/likelihood).
- Code:
  - Imports (PyTorch or JAX, numpy, matplotlib, etc.).
  - Device setup (CPU/GPU).
  - Seed setting for reproducibility.

Student tasks: none yet, just reading and running.

Time estimate: 10–15 min (including discussion).

---

### 3.2 Section 1 – Dataset and Baseline Visualization

Type: mostly provided code + 1–2 tiny TODOs.

Goals:
- Get students comfortable with the dataset and shapes.
- Provide quick intuition about the data before building the VAE.

Contents:
- Provided:
  - Data loader for MNIST or Fashion-MNIST (prefer something small and grayscale).
  - Functions to:
    - Show a grid of images.
    - Normalize and possibly binarize the data (for BCE loss).
- Student TODOs:
  - Possibly one small function for normalizing images (e.g. converting to [0,1]).
  - A short exercise to print shapes and verify batch structure.

Time estimate: 10–15 min.

---

### 3.3 Section 2 – Model Definition (Encoder, Decoder, VAE Wrapper)

Type: mix of provided skeletons + non-trivial TODOs.

Goals:
- Have students implement the core architecture of a simple VAE.
- Reinforce their understanding of what the encoder and decoder compute.

Architecture choice:
- Keep it simple to train quickly:
  - Input: 28×28 grayscale images.
  - Latent dimension: d_z = 2 (for nice visualization) or 10 (for improved recon).
  - Encoder: small CNN or MLP (use whichever aligns with the rest of the course).
  - Decoder: symmetric structure mapping z back to image logits.

Contents:
- Provided:
  - `Encoder` class skeleton:
    - `__init__` with defined layers.
    - `forward(self, x)` with TODOs.
  - `Decoder` class skeleton:
    - `__init__` with defined layers.
    - `forward(self, z)` with TODOs.
  - `VAE` wrapper:
    - `forward(self, x)` should return `(recon_logits, mu, log_var)`.
- Student TODOs:
  - Implement the forward pass in `Encoder` to output `mu` and `log_var`.
  - Implement the forward pass in `Decoder` to map `z` to image logits.
  - Implement `VAE.forward` logic using the encoder, reparameterization (hooked up later), and decoder.

Expected difficulty:
- Mostly routine neural network code; conceptually straightforward.

Time estimate: 30–40 min.

---

### 3.4 Section 3 – Reparameterization and VAE Loss

Type: core conceptual coding.

Goals:
- Implement the reparameterization trick in code.
- Implement the ELBO loss (reconstruction + KL).

Contents:
- Provided:
  - Function stub:
    ```python
    def reparameterize(mu, log_var):
        # TODO: implement reparameterization trick
        ...
    ```
  - Function stub:
    ```python
    def vae_loss(recon_logits, x, mu, log_var, beta=1.0):
        # TODO: compute reconstruction term and KL term
        ...
    ```
  - Notes on expected shapes and broadcasting.
- Student TODOs:
  - Implement `reparameterize` using:
    - `eps = torch.randn_like(mu)`
    - `z = mu + eps * torch.exp(0.5 * log_var)`
  - Implement reconstruction term:
    - Either BCE with logits or MSE; choose and clearly specify in markdown.
  - Implement analytic KL term for diagonal Gaussian vs standard normal:
    - `KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))` (per sample).
  - Combine into ELBO-based loss:
    - Loss = reconstruction + β * KL (where β=1 by default).

Support:
- Include markdown reminders of:
  - The formula for KL between two 1D Gaussians and its extension to diagonal.
  - Why the reparameterization trick works conceptually (without full proof).

Time estimate: 30–40 min (this is the main conceptual bottleneck).

---

### 3.5 Section 4 – Training Loop and Basic Evaluation

Type: provided training skeleton + a few lines to fill.

Goals:
- Train the VAE end-to-end.
- Make students comfortable with training loops and loss logging.

Contents:
- Provided:
  - Training loop skeleton:
    - Iteration over batches.
    - Zeroing gradients.
    - Logging structure (e.g. printing losses every N steps).
    - Optional validation loop structure.
- Student TODOs:
  - For each batch:
    - Call the VAE model to get `(recon_logits, mu, log_var)`.
    - Compute loss using `vae_loss`.
    - Backward pass and optimizer step.
  - Verify:
    - Loss decreases over epochs.
    - KL and reconstruction components behave reasonably.

Evaluation:
- Provided:
  - Functions to:
    - Plot original vs reconstructed images.
    - Sample from the prior and decode to images.
- Student tasks:
  - Run evaluation after training.
  - Answer short qualitative questions in markdown:
    - How do reconstructions compare to inputs?
    - Do samples look plausible?

Time estimate: 30–45 min (including running training, which should be short: a few minutes).

---

### 3.6 Section 5 – Latent Space Exploration and Interpolation

Type: mostly provided code + interpretive questions.

Goals:
- Help students **visualize the latent space**.
- Show interpolation and its connection to the learned manifold.

Contents:
- For low-dimensional latent (e.g. 2D):
  - Provided:
    - Code to encode a batch of images and plot `z` colored by class label.
  - Student tasks:
    - Run the plot.
    - Interpret clustering structure in 2D latent space.
- For interpolation:
  - Provided:
    - Function stub to interpolate between two z’s:
      ```python
      def interpolate(z1, z2, num_steps=10):
          # TODO: linearly interpolate in latent space
          ...
      ```
    - Code to decode each interpolated latent and plot the sequence.
  - Student TODOs:
    - Implement simple linear interpolation between z1 and z2.
    - Run and interpret results.

Reflection questions (markdown answers):
- Do interpolations correspond to smooth changes in the input space?
- How does latent dimension choice affect clustering and interpolation quality?

Time estimate: 20–30 min.

---

### 3.7 Section 6 – Optional Extensions and Hooks to Later Topics

Type: optional tasks for fast students; conceptual bridge to the rest of the course.

Possible extension tasks:
- β-VAE experiment:
  - Add a slider or config for β in the VAE loss.
  - Ask students to:
    - Train with β < 1 and β > 1.
    - Observe changes in reconstructions and latent structure.
- Latent dimension sweep:
  - Try d_z = 2 vs 10 vs 32.
  - Compare reconstructions and sample quality.

Hooks to hierarchical VAEs:
- Markdown explanation:
  - The current VAE has a **single-layer latent** `z`.
  - Hierarchical VAE will introduce `z₂` and `z₁`, with a generative structure like:
    - `z₂ ~ N(0, I)`
    - `z₁ ~ p(z₁ | z₂)`
    - `x ~ p(x | z₁)`
  - Emphasize:
    - The same reparameterization idea and KL terms reappear.
    - Only the structure of the latent space becomes richer.

Hooks to diffusion models:
- Markdown explanation:
  - In VAEs, latent `z` and KL regularization control how information is encoded.
  - Diffusion models:
    - Instead of a single latent, they work with noise added over many steps.
    - They also rely on understanding likelihoods, noise, and denoising behavior.
  - The intuition gained here (e.g. sampling from a prior, decoding to data space) will transfer to understanding the forward and reverse processes in diffusion.

Time estimate: variable; students who finish early can explore these; others can skip.

---

## 4. Calibration for a 4-Hour Session

For a typical student:
- Theory mini-talk (Section 2) + Section 0/1 reading: ~45–60 min.
- Sections 2–4 (core implementation and training): ~90–120 min.
- Section 5 (latent visualization & interpolation): ~20–30 min.
- Section 6 (optional extensions): remaining time for advanced students.

Key design constraints to enforce during implementation:
- **Short training time**:
  - Use small models and a limited number of epochs so that a full run fits in a few minutes on CPU.
- **Clear TODO boundaries**:
  - Mark all student-editable cells with `# TODO` and keep each TODO small (≤ 15–20 lines).
- **Embedded recap**:
  - Ensure all formulas needed for implementation appear in markdown cells right above their TODO.
- **Exam alignment**:
  - The code patterns used here (reparameterization, Gaussian KL, forward pass structure) should match the style to be used later in the exam notebooks, so the students see them multiple times.

This plan should now serve as the blueprint to implement `resources/VAE/01_VAE_practical.ipynb` (or similar), which will be the main entrance to the practical side of VAEs in the course and the foundation for the subsequent hierarchical VAE and diffusion sessions.

