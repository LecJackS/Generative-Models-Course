# One-Week Course Plan: VAEs and Diffusion Models (Practical Component)

Audience: Master’s and PhD students in ML / AI.  
Duration: 1 week, Monday–Friday (this document specifies only the practical part).  
Contact time for practical sessions: ~4–5 hours/day from Tuesday to Friday.  
Tools: Python, PyTorch/JAX (choose one), Jupyter/Colab-style notebooks, Git.

Scope:
- This plan focuses on practical, self-contained sessions: Tuesday–Thursday labs plus the practical part of the Friday exam.
- Theory lectures (probability, Markov chains/MCMC, diffusion-model proofs, etc.) are planned and delivered separately and may be asynchronous relative to the labs.
- Each notebook therefore includes its own short conceptual recap so students can follow even if the theory schedule drifts.

The emphasis is on conceptual understanding and the ability to read, modify, and debug implementations of VAEs and diffusion models under tight time constraints.

---

## 1. High-Level Course Objectives

By the end of the week, students should be able to:
- Explain VAEs, hierarchical VAEs, and their role in generative modeling.
- Understand and, when needed, derive the VAE ELBO and KL regularization (with support from separate theory lectures).
- Describe forward and reverse processes for continuous diffusion models (e.g. DDPM-style).
- Implement core pieces of a continuous diffusion model (noise schedules, loss, sampling).
- Describe and work with discrete diffusion models (e.g. on categorical/sequence data).
- Reason about tradeoffs (likelihood vs. sample quality, training stability, computation).
- Implement and debug missing pieces in provided code for VAEs and diffusion models under exam conditions.

---

## 2. Practical Components and Weekly Rhythm

Scope:
- This document does not design or depend on the theory-only classes; it assumes those may happen on different days or in a different order.
- The practical sessions (labs + practical exam) must stand on their own: any student with basic deep learning and probability should be able to complete them using only the notebooks.

Weekly rhythm for the *practical* part:
- Tuesday: VAEs and hierarchical VAEs – self-contained lab (Notebook 1–2).
- Wednesday: Continuous diffusion models – self-contained lab (Notebook 3–4).
- Thursday: Discrete diffusion models + integrated review – self-contained lab (Notebook 5–6).
- Friday: Final exam – practical notebooks (designed here) plus a theoretical written component coordinated with the separate theory instructors.

Interaction with separate theory lectures:
- When theory lectures align well with the labs, the conceptual recap segment at the start of each session can be shortened.
- When theory is asynchronous, students can still rely on the recap sections and inline explanations to bridge gaps.

---

## 3. Design Pattern for Practical Sessions (Tuesday–Thursday)

General pattern per practical day (4–5 hours):
- 30–45 min: Self-contained conceptual on-ramp + notebook walkthrough (high-level, inside the lab).
- 2–3 hours: Guided coding in notebooks (students work, instructor circulates).
- 30–45 min: Discussion, extensions, and Q&A.

Pedagogical constraints:
- Keep training times short: use small models and small datasets (e.g. MNIST, Fashion-MNIST, tiny CIFAR-10 subsets, synthetic 2D data).
- Pre-implement heavy/boilerplate code (data loading, plotting, training loop scaffolding).
- Students fill in critical conceptual pieces (loss functions, model components, key sampling steps).
- Do not assume students attended the theory lecture right before the lab or in a specific order; reintroduce any new object (e.g. KL for Gaussians, q(x_t | x_0)) briefly in the notebook itself.
- Allow (and encourage) ChatGPT use for:
  - Debugging syntax and simple logic.
  - Getting alternative explanations of concepts already introduced.
  - Suggesting small code snippets.
  - BUT: emphasize they should still understand and be able to reproduce derivations by hand for Friday.

---

## 4. Tuesday – VAEs and Hierarchical VAEs (Practical)

### Learning Goals
- Implement a basic VAE on images.
- Understand and code the reparameterization trick.
- Implement the VAE loss (reconstruction + KL).
- Extend to a simple hierarchical VAE and interpret the two-layer latent structure.

### Time Breakdown (4–5h)
- 0:00–0:30 – Self-contained conceptual on-ramp + notebook walkthrough
  - VAE architecture and training pipeline, recapped in notebook markdown cells (no dependency on a separate theory lecture).
  - KL divergence for Gaussian posteriors and its role in regularizing the latent space.
  - Connection between ELBO and reconstruction quality (intuitive explanation, minimal algebra).
  - Brief outline of the labs and expected outcomes.
- 0:30–2:00 – Notebook 1: Vanilla VAE (Core)
- 2:00–2:10 – Short break
- 2:10–3:40 – Notebook 2: Hierarchical VAE (Extension)
- 3:40–4:10 – Wrap-up:
  - Discussion of latent interpolations, reconstructions.
  - Common issues: posterior collapse, KL weighting.
  - Preview of diffusion models practicals.

### Notebook 1: Vanilla VAE on MNIST

Target: end-to-end training of a small VAE on MNIST (or similar). Students should:
- Implement the encoder and decoder networks.
- Implement the reparameterization trick.
- Implement the VAE loss and train the model.
- Visualize samples and latent interpolations.

Structure:
0. **Section 0 – Self-contained VAE recap (Markdown + small demo)**
   - Provided: short textual recap of the VAE objective, the ELBO (high level), and the reparameterization trick, written to be understandable without attending a separate theory lecture.
   - Optional: small pre-trained VAE model with a few visualizations to give intuition before coding.
1. **Section A – Setup**
   - Provided: imports, device setup, data loader, simple MLP/CNN skeleton.
   - Provided: code to visualize batches of images.
2. **Section B – Model Definition**
   - Provided:
     - `Encoder` and `Decoder` class templates with most boilerplate.
   - Student tasks:
     - Implement `Encoder.forward` to output `mu` and `log_var`.
     - Implement `Decoder.forward` to map latent z to image logits.
     - Implement `sample_z(mu, log_var)` using the reparameterization trick.
3. **Section C – Loss Function**
   - Provided:
     - Stub for `vae_loss(recon_logits, x, mu, log_var)`.
   - Student tasks:
     - Implement reconstruction term (e.g. BCE with logits).
     - Implement KL term for Gaussian posterior vs. standard normal prior.
     - Combine them into the ELBO (possibly with a KL weight parameter).
4. **Section D – Training Loop**
   - Provided: full training loop skeleton with comments, logging, and plotting.
   - Student tasks:
     - Call the model correctly.
     - Compute loss via their `vae_loss`.
     - Backprop and optimizer step (few lines).
5. **Section E – Evaluation & Visualization**
   - Provided: helper functions to:
     - Sample from the prior.
     - Interpolate between two latent vectors.
   - Student tasks:
     - Run the sampling code; interpret results.
     - Answer short questions (in markdown cells):
       - How does increasing KL weight affect samples?
       - How do reconstructions compare to original images?

### Notebook 2: Simple Hierarchical VAE

Target: conceptual understanding of hierarchical latent structure, not state-of-the-art performance.

Structure:
0. **Section 0 – Self-contained hierarchical VAE recap (Markdown)**
   - Provided: explanation of the idea of multiple latent layers (z_2, z_1), a diagram, and the factorization of the generative and inference models, written so it can be understood without the separate theory class.
1. **Section A – Motivation & Architecture**
   - Short markdown explanation of a 2-layer latent model:
     - z_2 ~ N(0, I)
     - z_1 ~ p(z_1 | z_2)
     - x ~ p(x | z_1)
   - Diagram included.
2. **Section B – Generative and Inference Models**
   - Provided:
     - Skeleton classes for `GenerativeModel` and `InferenceModel`.
   - Student tasks:
     - Implement mapping from z_2 → parameters of z_1 (e.g. Gaussian).
     - Implement encoder mappings from x → (z_1 params, z_2 params).
3. **Section C – ELBO for Hierarchical VAE**
   - Provided:
     - Outline of ELBO with comments.
     - Helper to compute KLs between Gaussians.
   - Student tasks:
     - Fill in terms for:
       - `E_q[log p(x|z_1)]`
       - `E_q[log p(z_1|z_2)]`
       - `E_q[log p(z_2)]`
       - `E_q[log q(z_1|x)]` and `E_q[log q(z_2|x)]`
     - Put together a numerical estimator of the ELBO using reparameterization.
4. **Section D – Training & Analysis**
   - Provided:
     - Training loop stub similar to Notebook 1.
   - Student tasks:
     - Train a small hierarchical VAE.
     - Visualize samples and evaluate whether the second latent layer captures global structure.
   - Short reflective questions:
     - Compare performance vs single-layer VAE.
     - What might hierarchical latents help with in more complex datasets?

---

## 5. Wednesday – Continuous Diffusion Models (Practical)

### Learning Goals
- Understand the discrete-time DDPM-style formulation in practice.
- Implement the forward noising process `q(x_t | x_0)` and sampling from it.
- Implement the diffusion loss (predicting noise epsilon or x_0).
- Use a small U-Net-like network or MLP to denoise.
- Generate samples via a simple reverse sampling loop.

### Time Breakdown (4–5h)
- 0:00–0:30 – Self-contained conceptual on-ramp + notebook walkthrough
  - From an intuitive random-noise-adding process to practical DDPM implementation (without relying on prior SDE/score proofs).
  - Beta schedules, alphas, and cumulative products, with diagrams and simple formulas in the notebook.
  - Parameterization choices and training objective (predicting noise vs x_0) at a practical level.
- 0:30–2:15 – Notebook 3: 1D/Toy DDPM (from-scratch intuition builder)
- 2:15–2:25 – Short break
- 2:25–4:00 – Notebook 4: Image DDPM with pre-written training loop
- 4:00–4:15 – Wrap-up and Q&A

### Notebook 3: Diffusion on 1D / 2D Toy Data

Target: build intuition with a very fast training setup (e.g. mixture of Gaussians or 2D Swiss roll).

Structure:
0. **Section 0 – Self-contained diffusion recap (Markdown + visual intuition)**
   - Provided: explanation of the forward noising process q(x_t | x_0), the idea of a reverse denoising model, and the training objective, including plots of increasingly noisy data.
1. **Section A – Setup & Data**
   - Provided: simple synthetic dataset (2D points).
   - Provided: plotting utilities to visualize data and diffusion steps.
2. **Section B – Forward Process**
   - Provided:
     - Beta schedule (e.g. linear).
     - Stub functions for:
       - `compute_alpha_cumprod(betas)`
       - `q_sample(x0, t, noise)`
   - Student tasks:
     - Implement `alpha`, `alpha_bar` computations.
     - Implement `q_sample` for sampling x_t given x_0 and a timestep t.
     - Visualize increasingly noisy x_t for different t.
3. **Section C – Model and Loss**
   - Provided:
     - Simple MLP model that takes (x_t, t_embed) and predicts noise epsilon.
     - Stub for loss function `diffusion_loss(model, x0, t)`.
   - Student tasks:
     - Implement time embedding (e.g. sinusoidal or simple learned embedding).
     - Implement `diffusion_loss`:
       - Sample t uniformly.
       - Sample noise epsilon.
       - Compute x_t via `q_sample`.
       - Predict epsilon_hat and compute MSE loss.
4. **Section D – Training & Sampling**
   - Provided:
     - Training loop skeleton.
     - Sampling loop stub `p_sample_loop`.
   - Student tasks:
     - Implement reverse sampling:
       - Start from x_T ~ N(0, I).
       - For t = T...1, compute mean and add noise (simple DDPM formula).
     - Generate and visualize final samples.
   - Short reflection:
     - Compare generated samples to original distribution.
     - How does number of steps T affect quality and runtime?

### Notebook 4: Simple Image DDPM (MNIST / CIFAR-10 subset)

Target: transfer the toy diffusion understanding to images with minimal boilerplate.

Structure:
0. **Section 0 – Image-specific diffusion recap (Markdown)**
   - Provided: brief restatement of the diffusion process for images (linking back to Notebook 3 concepts) and an explanation of how the network now operates on image tensors instead of low-dimensional points.
1. **Section A – Setup & Model**
   - Provided:
     - Data loaders for MNIST or small 32×32 dataset.
     - Small U-Net-like model skeleton (or simplified CNN).
   - Student tasks:
     - Fill in a few key components of the network (e.g. residual blocks or skip connections).
2. **Section B – Forward Process & Loss (Reuse from Notebook 3)**
   - Provided:
     - Beta schedule and q_sample functions (possibly identical to Notebook 3).
   - Student tasks:
     - Plug in image tensors.
     - Adapt loss function if necessary (e.g. per-pixel MSE).
3. **Section C – Training & Sampling**
   - Provided:
     - Training loop with logging but some gaps.
   - Student tasks:
     - Complete loss computation and backward pass.
     - Implement sampling loop (or adapt from Notebook 3).
   - Short reflection:
     - View generated samples.
     - Discuss differences in difficulty vs toy data.

---

## 6. Thursday – Discrete Diffusion Models (Practical)

### Learning Goals
- Understand diffusion over discrete states (e.g. categorical tokens).
- Implement a simple discrete diffusion process (e.g. masking or replacing tokens).
- Train a model to denoise discrete-corrupted sequences.
- Connect discrete diffusion to language modeling / masked modeling intuitions.
- Prepare students for Friday’s exam by revisiting core diffusion concepts.

### Time Breakdown (4–5h)
- 0:00–0:30 – Self-contained conceptual on-ramp + notebook walkthrough
  - Discrete vs continuous diffusion, explained with simple examples (e.g. tokens, characters).
  - Typical corruption processes (masking, random replacement) with explicit probability tables.
  - Training objectives and relation to cross-entropy, with at least one worked example in the notebook.
- 0:30–2:30 – Notebook 5: Discrete Diffusion on Toy Sequences
- 2:30–2:40 – Short break
- 2:40–3:40 – Notebook 6: Integrated Mini-Project / Review
- 3:40–4:10 – Review and exam briefing

### Notebook 5: Discrete Diffusion on Short Sequences

Possible setting: sequences of tokens drawn from a small vocabulary (e.g. digits, small words, or quantized image patches).

Structure:
0. **Section 0 – Self-contained discrete-diffusion recap (Markdown + simple example)**
   - Provided: intuitive description of discrete corruption (e.g. progressively masking characters in a word) and how a model learns to reverse it.
1. **Section A – Data & Tokenization**
   - Provided:
     - Simple dataset of token sequences.
     - Vocabulary and embedding lookup.
2. **Section B – Forward Discrete Corruption Process**
   - Provided:
     - Stub for `q_sample_discrete(x0, t)` which progressively replaces tokens with MASK or random tokens according to a schedule.
   - Student tasks:
     - Implement the time-dependent corruption probabilities.
     - Verify behavior by visualizing sequences at different t.
3. **Section C – Denoising Model**
   - Provided:
     - Simple Transformer or bidirectional RNN skeleton.
   - Student tasks:
     - Implement forward pass to produce logits over vocabulary for each position.
4. **Section D – Training Objective**
   - Provided:
     - Stub for `discrete_diffusion_loss(model, x0, t)`.
   - Student tasks:
     - Sample t.
     - Corrupt x0 to get x_t.
     - Predict original tokens and compute cross-entropy loss.
5. **Section E – Sampling / Denoising**
   - Provided:
     - Sampling loop stub.
   - Student tasks:
     - Implement iterative denoising from fully corrupted sequence back to clean.
     - Examine qualitative results (e.g. how well model recovers tokens).

### Notebook 6: Integrated Mini-Project / Review

Target: consolidate understanding and mimic exam-style tasks without grading pressure.

Structure:
1. **Section A – Conceptual Review Questions (Markdown)**
   - Short written prompts for students to answer in their own words:
     - Compare VAEs and diffusion models in terms of:
       - Training objectives.
       - Latent structure.
       - Sample quality vs. likelihood.
     - Explain the role of the KL term in VAEs and the noise schedule in diffusion models.
2. **Section B – Small Coding Tasks (Exam-like)**
   - Provide 3–4 small self-contained coding problems, such as:
     - Implement a Gaussian KL function (used in VAEs).
     - Implement the reparameterization trick for a given encoder.
     - Implement one step of the diffusion reverse update.
     - Implement the discrete corruption layer for a given t.
   - Each task should:
     - Fit on a single screen.
     - Have clear input/output signatures and unit tests students can run.
3. **Section C – Exam Preparation Guidelines**
   - Markdown cell describing:
     - How Friday’s exam will be structured.
     - Example types of questions (but not actual exam questions).
     - Reminder that ChatGPT will not be allowed and that they should practice doing derivations and small coding tasks without assistance.

---

## 7. Friday – Final Exam (Theoretical + Practical)

### Constraints and Logistics
- Duration: 3–4 hours.
- Closed-book for ChatGPT and external internet; allow:
  - Local environment, standard docs (if allowed by institution).
  - Any course materials (slides, notebooks) depending on policy.
- Students work individually.

### Exam Structure (Example Allocation)
- 40% Theoretical (written):
  - Short-answer and derivation questions.
- 60% Practical (coding in a notebook):
  - Implement missing pieces in diffusion-related code.

### Theoretical Part (40%)

The detailed content of the theoretical questions should be designed by, or at least closely coordinated with, the instructors responsible for the separate theory lectures (probability, MCMC, proofs for diffusion models, etc.).  
High-level guidance:
- Cover VAEs, hierarchical VAEs, continuous diffusion, and discrete diffusion at a level consistent with what was proven in the theory sessions.
- Emphasize connections to the practical labs (e.g. ELBO terms, forward vs reverse diffusion, discrete corruption processes), but do not assume a specific order in which students encountered these topics.

Assessment focus:
- Correctness of derivations.
- Clarity in explaining intuitions.
- Ability to connect different models conceptually.

### Practical Part (60%)

One or two exam notebooks; students implement missing parts. ChatGPT is not allowed, so tasks must be doable based on what they practiced.

**Exam Notebook A – Continuous Diffusion Mini-Task**
- Given:
  - Data loader for a tiny image dataset.
  - Predefined model architecture.
  - Beta schedule skeleton.
- Tasks:
  - Implement `q_sample(x0, t, noise)` (forward process).
  - Implement `diffusion_loss(model, x0, t)` for epsilon prediction.
  - Implement a simple reverse sampling loop from x_T to x_0.
  - Each coding task is accompanied by a short recap markdown cell restating the relevant formulas (without giving away the solution), so the exam remains self-contained with respect to the practical content.
- Tests:
  - Provide simple unit tests to check shapes and basic consistency (students run but do not see solutions).

**Exam Notebook B – Discrete Diffusion Mini-Task (Optional or second part)**
- Given:
  - Small token-sequence dataset.
  - Model skeleton (RNN/Transformer).
  - Corruption schedule description.
- Tasks:
  - Implement discrete corruption function `q_sample_discrete(x0, t)`.
  - Implement cross-entropy-based denoising loss.
  - Implement 1–2 steps of the denoising process during sampling.
  - As above, each task includes a brief restatement of the discrete corruption and denoising setup to keep the practical exam self-contained.

Grading criteria:
- Correct implementation of core functions.
- Code runs without errors on provided tests.
- Reasonable sample quality or at least consistent behavior given limited time.

---

## 8. Use of ChatGPT and AI Assistants

- Tuesday–Thursday:
  - Allowed and encouraged for:
    - Syntax help, debugging, and code cleanup.
    - Alternative explanations of concepts already covered.
    - Suggestions for extensions or experiments (e.g. trying different architectures).
  - Not allowed for:
    - Directly asking for solutions to graded components (if any).
    - Copy-pasting large unmodified answers without understanding.
- Friday (Exam):
  - Strictly prohibited.
  - Students should rely on their understanding and the patterns from the week.

Recommendations:
- Make this explicit at the start of the week and repeat on Thursday.
- Encourage students to use ChatGPT as a “tutor” rather than a “solution generator”.
- Consider including a small reflection question about how they used AI during the week (not graded, just feedback).

---

## 9. Implementation Notes for Instructor

- Environment:
  - Prepare a unified environment (e.g. Conda env or Docker) with:
    - `torch` or `jax`, `numpy`, `matplotlib`, `tqdm`, etc.
  - Test all notebooks on the target machines to ensure:
    - Training completes within minutes (not hours).
    - No large downloads are triggered during class.
- Datasets:
  - Prefer small datasets or small subsets to keep training time low.
  - Pre-download and cache to avoid network issues.
- Notebooks:
  - Ensure each notebook:
    - Has a clear “Goals” section at the top.
    - Clearly marks student-editable cells (e.g. with `# TODO`).
    - Includes simple sanity checks (assertions, unit tests) where appropriate.
- Time management:
  - Include “checkpoint” cells so students who fall behind can jump to later sections if needed.
  - Provide solution branches after the course (not during).

This plan should be used as a blueprint to build the actual notebooks under the repository (e.g. `resources/VAE`, `resources/diffusion_continuous`, `resources/diffusion_discrete`), with each section above corresponding to one or more notebooks.
