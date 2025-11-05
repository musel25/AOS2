# Demonstration: stacking multiple **linear** layers (no activations) 
# is exactly equivalent to a single affine (linear + bias) transformation.

import numpy as np

rng = np.random.default_rng(42)

# Dimensions
n_samples = 8      # batch size
d_in = 5           # input dimension
hidden = [7, 6, 4] # three "hidden" linear layers (no activations)
d_out = 3          # output dimension

# Generate dummy input data
X = rng.normal(size=(n_samples, d_in))


# Build random linear layers (weights and biases)
def random_layer(in_dim, out_dim, scale=0.5):
    W = rng.normal(scale=scale, size=(in_dim, out_dim))
    b = rng.normal(scale=scale, size=(out_dim,))
    return W, b

dims = [d_in] + hidden + [d_out]
Ws, bs = [], []
for i in range(len(dims)-1):
    W, b = random_layer(dims[i], dims[i+1])
    Ws.append(W)
    bs.append(b)

# Forward pass through the stack of linear layers (no activations)
def forward_stack(X, Ws, bs):
    H = X
    for W, b in zip(Ws, bs):
        H = H @ W + b  # affine map
    return H

Y_stack = forward_stack(X, Ws, bs)

# Collapse the stack into a single affine map: Y = X @ W_eff + b_eff
def collapse_affine(Ws, bs):
    W_eff = Ws[0].copy()
    b_eff = bs[0].copy()
    for W, b in zip(Ws[1:], bs[1:]):
        # After previous layers we have: H = X @ W_eff + b_eff
        # Next layer outputs: H @ W + b = (X @ W_eff + b_eff) @ W + b
        #                   = X @ (W_eff @ W) + (b_eff @ W + b)
        b_eff = b_eff @ W + b
        W_eff = W_eff @ W
    return W_eff, b_eff

W_eff, b_eff = collapse_affine(Ws, bs)
Y_eff = X @ W_eff + b_eff

# Compare results
max_abs_diff = np.max(np.abs(Y_stack - Y_eff))

print("Shapes:")
print(f"  X:      {X.shape}")
for i, (W, b) in enumerate(zip(Ws, bs), 1):
    print(f"  Layer {i}  W: {W.shape},  b: {b.shape}")
print(f"  Collapsed W_eff: {W_eff.shape}, b_eff: {b_eff.shape}\n")

print("First 3 rows of Y from stacked linear layers:\n", Y_stack[:3])
print("\nFirst 3 rows of Y from single affine map:\n", Y_eff[:3])
print(f"\nMax absolute difference between the two outputs: {max_abs_diff:.3e}")

# Sanity check (should be ~ 1e-15 to 1e-12 due to floating point)
assert np.allclose(Y_stack, Y_eff, atol=1e-10), "Mismatch! The equivalence failed."

print("\nConclusion: Multiple linear layers without activations are exactly one affine map.")
