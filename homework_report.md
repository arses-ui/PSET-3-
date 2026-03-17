# ENGS 106 - Assignment 3 Report (Parts 3A, 3B, 3C)

Name: Arses Prasai
Date: feb 15  

## Part 3A - Sparse Autoencoder

### A. Model description and parameter choices
I implemented a single-hidden-layer sparse autoencoder for 8x8 grayscale patches sampled from the provided natural images dataset.

- `patchsize = 8`, so `visibleSize = 64`
- `hiddenSize = 25`
- `sparsityParam = 0.01`
- `beta = 3`
- `decayWeight = 0.0001`
- `numpatches = 10000`

These settings match the lab defaults and are appropriate because the hidden layer is overcomplete enough to learn edge-like filters while sparsity keeps most hidden activations near zero.

### B. Implementation-required questions
1. `sampleIMAGES()`
- Randomly samples 10,000 patches.
- For each sample: pick one of 10 images, pick a random top-left coordinate, extract an 8x8 patch, flatten to length 64.
- Applies normalization to map values into `[0.1, 0.9]`.

2. `sparseAutoencoderCost(...)`
- Implemented forward pass:
  - `a2 = sigmoid(W1^T x + b1)`
  - `a3 = sigmoid(W2^T a2 + b2)`
- Cost includes:
  - Reconstruction squared error
  - Weight decay regularization
  - KL sparsity penalty
- I `Implemented backprop to compute `W1grad`, `W2grad`, `b1grad`, `b2grad`.

3. `computeNumericalGradient(J, theta)`
- Implemented centered finite difference:
  - `numgrad[i] = (J(theta+eps*e_i) - J(theta-eps*e_i)) / (2*eps)`
- Used `eps = 1e-4`.

4. `predict(self, samples)`
- I Implemented normal forward propagation through hidden and output layers and returned reconstructed outputs.

### C. Evaluation
Observed notebook outputs:

- Untrained score: `48.84739443091225`
- Trained score: `0.45183695641617005`
- L1 reconstruction error: `7.081297929995559`

Gradient-check sanity result (from validation run):

- Relative difference: `2.1452381569477388e-12` (very small, indicates correct gradient implementation).

The trained model significantly reduced reconstruction cost, and filter visualizations showed structured learned features.

### D. Reflection
This part was pretty useful for understanding how sparsity and KL regularization change hidden-layer behavior compared to a plain autoencoder. The gradient check was important for debugging and playing around more with the backpropgation code and gave confidence that backprop was implemented correctly.

---

## Part 3B - Transformer Language Model (Shakespeare)

### A. Model description and parameter choices
I implemented a GPT-style decoder-only Transformer for character-level text generation.

Main configuration used:

- `context_length = 256`
- `n_layer = 6`
- `n_head = 6`
- `n_embd = 384`
- `dropout = 0.2`
- `learning_rate = 1e-3`
- `batch_size = 64`
- `max_iters = 2000`

This setup balances model capacity and runtime. Six layers and six heads should be enough to learn local and medium-range token dependencies in Shakespeare text without making training too slow.

### B. Implementation-required questions
1. `Attention.attention(self, q, k, v, T)`
- Implemented scaled dot-product attention:
  - `weights = (q @ k^T) / sqrt(d_k)`
  - applied causal mask
  - `attention = dropout(softmax(weights)) @ v`

2. `MultiHeadAttention.forward(self, x)`
- Ran each head independently.
- Concatenated all head outputs along the embedding dimension.
- Applied output projection and dropout.

3. `PositionalEncoding.__init__`
- Filled sinusoidal positional encoding matrix:
  - even indices: `sin(...)`
  - odd indices: `cos(...)`
- Registered encoding as a non-trainable buffer.

### C. Evaluation
Observed notebook outputs:

- Total trainable parameters: `10690625`
- Sample generated text begins as:

> ROMEO:  
> Romeo, thou hast not that thou makest thy breath,  
> Nor nothing an agreet to thy party sen ...

You can see that the generated sample is not perfect English, but it captures style-like structure (speaker tags, line breaks, archaic wording), which is expected for a small model and limited training.

### D. Reflection
This part clarified how attention, masking, and positional encodings combine in a working autoregressive model. I also learned that qualitative generation can already look stylistically meaningful even before full convergence.

---

## Part 3C - Diffusion Model on MNIST

### A. Model description and parameter choices
I implemented a simplified DDPM with a U-Net backbone for MNIST generation.

Configuration:

- `batch_size = 128`
- `learning_rate = 2e-4`
- `num_epochs = 10`
- `num_timesteps = 1000`
- `beta_start = 1e-4`, `beta_end = 0.02`
- `hidden_dims = 64`

The linear noise schedule and 1000 diffusion steps are standard DDPM choices. A moderate U-Net width (`hidden_dims=64`) keeps the model expressive while still runnable on CPU.

### B. Implementation-required questions
1. `NoiseScheduler.__init__`
- Implemented:
  - linear `betas`
  - `alphas = 1 - betas`
  - cumulative products `alpha_cumprods`
  - square-root helper tensors for forward diffusion.

2. `SinusoidalPositionEmbedding.forward`
- Implemented sinusoidal timestep embeddings using sin/cos pairs.

3. `SimpleUNet.forward`
- Implemented full encoder-middle-decoder flow:
  - time embedding
  - down blocks with pooling and skip connections
  - middle block
  - upsampling + concatenation + conv blocks
  - final 1x1 projection.

4. `train_one_epoch`
- Sampled random timesteps per batch.
- Added noise via scheduler.
- Predicted noise with U-Net.
- Minimized MSE loss vs true noise.

5. `sample_step`
- Implemented reverse DDPM update:
  - predict noise
  - compute denoised mean
  - add stochastic term for `t > 0`
  - return mean at final step.

### C. Evaluation
Observed notebook outputs:

- Device: `cpu`
- Dataset size: `60000`
- Number of batches: `469`
- Total trainable parameters: `1,011,969`

Training loss by epoch:

- Epoch 1: `0.0773`
- Epoch 2: `0.0373`
- Epoch 3: `0.0332`
- Epoch 4: `0.0311`
- Epoch 5: `0.0295`
- Epoch 6: `0.0285`
- Epoch 7: `0.0278`
- Epoch 8: `0.0274`
- Epoch 9: `0.0265`
- Epoch 10: `0.0262`

As you can see, loss decreases steadily, indicating successful learning of the denoising objective. Generated-sample and trajectory figures show denoising progression from noise to digit-like outputs.

### D. Reflection
This part made the forward/reverse diffusion math concrete for me in code. Implementing sampling step-by-step gave me a much clearer understanding of how generation emerges from iterative denoising.

---

## Overall Takeaways
Across all three parts, I think the key theme was implementing core generative-model mechanics directly:

- Sparse coding and regularized reconstruction (Autoencoder)
- Sequence modeling with masked attention (Transformer)
- Iterative denoising for generation (Diffusion)

The assignment connected the math to practical PyTorch implementations and highlighted how debugging tools (especially gradient checking and simple sanity runs) are critical for correctness.
