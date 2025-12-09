# Discrete Diffusion (D3PM) – Teaching Plan and Notebook Blueprint

Use this as the blueprint for the discrete diffusion (D3PM) notebook. It assumes students just saw continuous diffusion; emphasize similarities and key differences. Audience: strong probability background, hands-on focus, same-day delivery.

---

## Resources to leverage
- `lecture16-2023-comp.pptx`, `cs236_lecture17.pdf`, `cs236_lecture18.pdf`: theory slides on discrete diffusion/D3PM.
- `My notes on discrete denoising diffusion models (D3PMs) _ Christopher Beckham, PhD.html`: concise conceptual notes and pitfalls.
- `d3pm-main/` minimal PyTorch implementation:
  - `d3pm_runner.py`: end-to-end MNIST D3PM with hybrid loss, uniform corruption matrix, sampler, conditional UNet-like model.
  - `dit.py`, `lm.py`: alternative model variants (DiT, language model style).
  - `contents/*.gif/png`: visual sample trajectories.
  - `test.ipynb`: scratch tests for logits/posteriors.
- Paper: Austin et al., “Structured Denoising Diffusion Models in Discrete State-Spaces”.

---

## Learning goals (explicit for students)
- Translate continuous diffusion intuition to discrete/categorical settings.
- Construct forward noising via transition matrices \(Q_t\) and cumulative \(Q_{1:t}\).
- Derive and implement the exact posterior \(q(x_{t-1}\mid x_t, x_0)\); understand Gumbel-max sampling.
- Implement and sanity-check the hybrid loss (VB term + CE on \(x_0\) logits).
- Train a tiny D3PM on toy/MNIST data; inspect samples; understand class conditioning.
- Compare discrete vs. continuous diffusion (commonalities/differences; when to use which).

---

## Notebook structure (step-by-step)

### 0) Orientation
- Short recap of continuous diffusion; contrast with discrete:
  - State space: categorical vs. continuous.
  - Forward: categorical transition matrices (vs. Gaussian noise).
  - Reverse: predict \(x_0\) logits → posterior over \(x_{t-1}\).
- “How to use this notebook today” checklist (run order, minimal path).

### 1) Discrete forward process (intuition + code)
- Tiny K-class toy (e.g., K=3 colors or digits mod 3).
- One-step corruption \(Q^{(t)}\) (start with uniform corruption).
- Build cumulative \(Q_{1:t}\); show it flattens with t.
- Code: construct \(Q^{(t)}\), \(Q_{1:t}\); visualize row-stochastic matrices; apply to one-hot batches.

### 2) Posterior \(q(x_{t-1}\mid x_t, x_0)\)
- Derive Eq. (3) from D3PM: combine \(Q^{(t)}\) row (from \(x_t\)) and \(Q_{1:t-1}\) column (from \(x_0\)).
- Explain Gumbel-max for categorical sampling from logits.
- Code: compute posterior logits given \(x_0, x_t, t\); sample with Gumbel; compare empirical to analytic on a toy batch.

### 3) Loss: hybrid VB + CE
- Two parts: variational bound term (posterior matching) + CE on \(x_0\) logits.
- Why CE on \(x_0\) stabilizes training; role of hybrid coefficient.
- Code: implement `vb` and hybrid loss on toy logits; check shapes/values.

### 4) Minimal model and training loop (small/fast)
- Simplify UNet from `d3pm_runner.py` (fewer channels) for in-class runtime.
- Dataset: MNIST discretized to N bins (start with N=2 or 4).
- Training loop: few epochs, small batch; log CE/VB; plot loss curves.
- Checkpoints: expected loss scale; what to do if NaN (clip logits, check Q shapes).

### 5) Sampling
- Reverse sampling loop (`p_sample`/`sample_with_image_sequence`); keep steps small for speed.
- Generate small grids; optional GIF (reduced stride/steps).
- Conditional sampling: show label embedding injection (as in `DummyX0Model`).

### 6) Interactive probes
- Sliders for t to visualize:
  - Forward corruption of a sample image.
  - Posterior over \(x_{t-1}\) given \(x_t\) and guessed \(x_0\).
- Toggle uniform vs. skewed corruption to see sampling impact (optional).

### 7) Connections and comparisons
- Commonalities: forward noising + learned reverse, time-dependent noise schedule.
- Differences: reparameterization (Gumbel vs. Gaussian), transition structure, data domains.
- When to use discrete diffusion (text/tokens, quantized codes, color bins) vs. continuous.

### 8) Pitfalls / debug checklist
- Verify \(Q\) shapes; rows sum to 1.
- Broadcasting/dtypes in `q_posterior_logits`; float32 vs. float64.
- Gumbel noise clip away from 0/1 to avoid infs.
- CE explosions → lower LR, reduce num_classes/bins, or hybrid weight tuning.

### 9) Stretch goals (optional)
- Swap forward schedule (non-uniform betas).
- Try DiT backbone (`dit.py`) or `lm.py` for token diffusion (out of scope for class runtime).
- Conditioned corruption matrices; explore class-conditional corruption.

---

## Inline Q&A prompts (hide answers with `<details>`)
- Why does the posterior combine a guess from \(x_t\) and from \(x_0\)?
- Why is CE on \(x_0\) helpful beyond the VB term?
- How does num_classes/binning affect reconstruction vs. sharpness?

---

## Reuse plan for existing code
- From `d3pm_runner.py`: reuse simplified UNet (`DummyX0Model`), `D3PM` (forward/posterior/sampler), training loop; trim channels/epochs for class use.
- From `test.ipynb`: quick logits/posterior sanity checks.
- From `contents/*.gif`: embed as “target” visuals after training.

