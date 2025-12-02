# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#     path: /usr/share/jupyter/kernels/python3
# ---

# %% [markdown]
# # Vision transformer
#
# ## Preliminaries
#
# ### Libraries and imports

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %% [markdown]
# ### Global variables

# %%
# MNIST images are 28x28
IMAGE_SIZE = 28

# Divide image into (28/7)x(28/7) patches
PATCH_SIZE = 7
NUM_SPLITS = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_SPLITS ** 2

BATCH_SIZE = ...
EMBEDDING_DIM = ...
NUM_HEADS = ...
NUM_CLASSES = ...
NUM_TRANSFORMER_LAYERS = ...
HIDDEN_DIM = ...
EPOCHS = ...
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## The `MNIST` dataset
#
# See [here](https://en.wikipedia.org/wiki/MNIST_database) for details.

# %%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root="./.data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./.data", train=False, transform=transform, download=True
)

# Define data loader with `BATCH_SIZE` and shuffle
train_loader = ...
test_loader = ...


# %% [markdown]
# ## Patch Embedding Layer
#
# The first module to implement is a module that will transformed a tensor
# of size `BATCH_SIZE` \* 1 \* `IMAGE_SIZE` \* `IMAGE_SIZE` into a tensor
# of size `BATCH_SIZE` \* `NUM_PATCHES` \* `EMBEDDING_DIM`. This can be
# done by using a `nn.Conv2d` module with both the stride and the kernel
# the size of a patch.

# %%
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=7, embedding_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        # Use `nn.Conv2d` to split the image into patches
        self.projection = ...

    def forward(self, x):
        # `x` is `BATCH_SIZE` * 1 * `IMAGE_SIZE` * `IMAGE_SIZE`

        # Project `x` into a tensor of size `BATCH_SIZE` * `EMBEDDING_DIM` *
        # `NUM_SPLITS` * `NUM_SPLITS`
        x = ...

        # Flatten spatial dimensions to have a tensor of size `BATCH_SIZE` *
        # `EMBEDDING_DIM` * `NUM_PATCHES`
        x = ...

        # Put the `NUM_PATCHES` dimension at the second place to have a tensor
        # of size `BATCH_SIZE` * `NUM_PATCHES`` * `EMBEDDING_DIM`
        x = ...

        return x


# %% [markdown]
# ## Transformer encoder

# %%
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        # Define a `nn.MultiheadAttention` module with `embedding_dim` and
        # `num_heads`. Don't forget to set `batch_first` to `True`
        self.attention = ...

        # Define the position-wise feed-forward network using an `nn.Sequential`
        # module, which consists of a linear layer, a GELU activation function,
        # and another linear layer
        self.mlp = ...

        # Define two layer normalization modules
        self.layernorm1 = ...
        self.layernorm2 = ...

    def forward(self, x):
        # Compute self-attention on `x`
        attn_output, _ = ...

        # Skip-connection and first layer normalization
        x = ...

        # Apply the position-wise feed-forward network
        mlp_output = ...

        # Skip-connection and second layer normalization
        x = ...

        return x


# %% [markdown]
# ## Vision Transformer

# %%
class VisionTransformer(nn.Module):
    def __init__(
            self,
            patch_size,
            embedding_dim,
            num_heads,
            num_classes,
            num_transformer_layers,
            hidden_dim,
    ):
        super().__init__()

        # Define a `PatchEmbedding` module
        self.patch_embedding = ...

        # Use `nn.Parameter` to define an additional token embedding that will
        # be used to predict the class
        self.cls_token = ...

        # Use `nn.Parameter` to define a learnable positional encoding.
        self.position_embedding = ...

        # Use `nn.init.xavier_uniform_` to initialize the positional embedding
        ...

        self.encoder_layers = ...

        self.mlp_head = ...

    def forward(self, x):
        # `x` is `BATCH_SIZE` * 1 * `IMAGE_SIZE` * `IMAGE_SIZE`

        # Transform images into embedded patches. It gives a tensor of size
        # `BATCH_SIZE` * `NUM_PATCHES` * `EMBEDDING_DIM`
        x = ...

        # We need to add the embedded classification token at the beginning of
        # each sequence in the minibatch. Use `expand` to duplicate it along the
        # batch size dimension
        batch_size = ...
        cls_tokens = ...

        # Next use `torch.cat` to concatenate `cls_tokens` and `x` to have a
        # tensor of size `BATCH_SIZE` * (NUM_PATCHES + 1) * `EMBEDDING_DIM`
        x = ...

        # Add the positional encoding
        x += ...

        # Apply the stacked transformer modules
        y = ...

        # Select the classification token for each sample in the minibatch.
        # `cls_output` should be of size `BATCH_SIZE` * 1 * `EMBEDDING_DIM`
        cls_output = ...

        # Use `self.mlp_head` to adapt the output size to NUM_CLASSES.
        out = ...

        return out


# %% [markdown]
# ## Initialize model, loss and optimizer

# %%
# Define the `VisionTransformer` model
model = ...

# Use cross-entropy loss and AdamW optimizer with a learning rate of 5e-3
criterion = ...
optimizer = ...


# %% [markdown]
# ## Validation loss calculation

# %%
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


# %% [markdown]
# ## Training with Validation

# %%
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate validation loss and accuracy
    val_loss, val_accuracy = validate_model(model, test_loader, criterion)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
