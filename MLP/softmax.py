import numpy as np
import matplotlib.pyplot as plt

# Sample logits vector
logits = np.array([2.0, 1.0, 0.1])

# Softmax function
def softmax(x):
    exps = np.exp(x - np.max(x))  # for numerical stability
    return exps / np.sum(exps)

# Compute probabilities
probs = softmax(logits)

# Plot logits and softmax probabilities
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot logits
axes[0].bar(range(len(logits)), logits)
axes[0].set_title("Logits (Input)")
axes[0].set_xticks(range(len(logits)))
axes[0].set_xlabel("Index")
axes[0].set_ylabel("Value")

# Plot softmax probabilities
axes[1].bar(range(len(probs)), probs)
axes[1].set_title("Softmax Probabilities (Output)")
axes[1].set_xticks(range(len(probs)))
axes[1].set_xlabel("Index")
axes[1].set_ylabel("Probability")

plt.tight_layout()
plt.show()
