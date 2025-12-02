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
# # Word embedding and RNN for sentiment analysis
#
# The goal of the following notebook is to predict whether a written
# critic about a movie is positive or negative. For that we will try three
# models. A simple linear model on the word embeddings, a recurrent neural
# network and a CNN.
#
# ## Preliminaries
#
# ### Libraries and Imports
#
# First some imports are needed.

# %%
from timeit import default_timer as timer
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

# %% [markdown]
# ### Global variables
#
# First let’s define a few variables. `EMBEDDING_DIM` is the dimension of
# the vector space used to embed all the words of the vocabulary.
# `SEQ_LENGTH` is the maximum length of a sequence, `BATCH_SIZE` is the
# size of the batches used in stochastic optimization algorithms and
# `NUM_EPOCHS` the number of times we are going thought the entire
# training set during the training phase.

# %%
EMBEDDING_DIM = ...
SEQ_LENGTH = ...
BATCH_SIZE = ...
NUM_EPOCHS = ...

# %% [markdown]
# ## The `IMDb` dataset
#
# We use the `datasets` library to load the `IMDb` dataset.

# %%
dataset = load_dataset("imdb")
train_set = dataset['train']
test_set = dataset['test']

train_set[0]

print(f"Number of training examples: {len(train_set)}")
print(f"Number of testing examples: {len(test_set)}")

# %% [markdown]
# ### Building a vocabulary out of `IMDb` from a tokenizer
#
# We first need a tokenizer that takes a text a returns a list of tokens.
# There are many tokenizers available from other libraries. Here we use
# the `tokenizers` library.

# %%
# Use a word-level tokenizer in lower case
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# %% [markdown]
# Then we need to define the set of words that will be understood by the
# model: this is the vocabulary. We build it from the training set.

# %%
train_texts = train_set['text']
test_texts = test_set['text']

trainer = trainers.WordLevelTrainer(vocab_size=10000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(train_texts, trainer)

vocab = tokenizer.get_vocab()

UNK_IDX, PAD_IDX = vocab["[UNK]"], vocab["[PAD]"]
VOCAB_SIZE = len(vocab)

tokenizer.encode("All your base are belong to us").tokens
tokenizer.encode("All your base are belong to us").ids

vocab['plenty']


# %% [markdown]
# ## The training loop
#
# The training loop is decomposed into 3 different functions:
#
# -   `train_epoch`
# -   `evaluate`
# -   `train`
#
# ### Collate function
#
# The collate function maps raw samples coming from the dataset to padded
# tensors of numericalized tokens ready to be fed to the model.

# %%
def collate_fn(batch: List):
    def collate(text):
        """Turn a text into a tensor of integers."""
        ids = tokenizer.encode(text).ids[:SEQ_LENGTH]
        return torch.LongTensor(ids)

    src_batch = [collate(sample["text"]) for sample in batch]

    # Pad list of tensors using `pad_sequence`
    src_batch = ...

    # Define the labels tensor
    tgt_batch = ...

    return src_batch, tgt_batch


# %% [markdown]
# ### The `accuracy` function
#
# We need to implement an accuracy function to be used in the
# `train_epoch` function (see below).

# %%
def accuracy(predictions, labels):
    # `predictions` and `labels` are both tensors of same length

    # Implement accuracy
    return ...

assert accuracy(torch.Tensor([1, -2, 3]), torch.Tensor([1, 0, 1])) == 1
assert accuracy(torch.Tensor([1, -2, -3]), torch.Tensor([1, 0, 1])) == 2 / 3


### The `train_epoch` function

def train_epoch(model: nn.Module, optimizer: Optimizer):
    model.to(device)

    # Training mode
    model.train()

    loss_fn = nn.BCEWithLogitsLoss()

    train_dataloader = DataLoader(
        train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    matches = 0
    losses = 0
    for sequences, labels in train_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        # Implement a step of the algorithm:
        #
        # - set gradients to zero
        # - forward propagate examples in `batch`
        # - compute `loss` with chosen criterion
        # - back-propagate gradients
        # - gradient step
        ...

        acc = accuracy(predictions, labels)

        matches += len(predictions) * acc

    return losses / len(train_set), matches / len(train_set)


# %% [markdown]
# ### The `evaluate` function

# %%
def evaluate(model: nn.Module):
    model.to(device)
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()

    val_dataloader = DataLoader(
        test_set, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    losses = 0
    matches = 0
    for sequences, labels in val_dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        predictions = model(sequences)
        loss = loss_fn(predictions, labels)
        acc = accuracy(predictions, labels)
        matches += len(predictions) * acc
        losses += loss.item()

    return losses / len(test_set), matches / len(test_set)


# %% [markdown]
# ### The `train` function

# %%
def train(model, optimizer):
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss, train_acc = train_epoch(model, optimizer)
        end_time = timer()
        val_loss, val_acc = evaluate(model)
        print(
            f"Epoch: {epoch}, "
            f"Train loss: {train_loss:.3f}, "
            f"Train acc: {train_acc:.3f}, "
            f"Val loss: {val_loss:.3f}, "
            f"Val acc: {val_acc:.3f}, "
            f"Epoch time = {(end_time - start_time):.3f}s"
        )


# %% [markdown]
# ### Helper function to predict from a character string

# %%
def predict_sentiment(model, sentence):
    "Predict sentiment of given sentence according to model"

    tensor, _ = collate_fn([("dummy", sentence)])
    prediction = model(tensor)
    pred = torch.sigmoid(prediction)
    return pred.item()


# %% [markdown]
# ## Models
#
# ### Training a linear classifier with an embedding
#
# We first test a simple linear classifier on the word embeddings.

# %%
class EmbeddingNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, seq_length):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Define an embedding of `vocab_size` words into a vector space
        # of dimension `embedding_dim`.
        self.embedding = ...

        # Define a linear layer from dimension `seq_length` *
        # `embedding_dim` to 1.
        self.l1 = ...

    def forward(self, x):
        # `x` is of size `seq_length` * `batch_size`

        # Compute the embedding `embedded` of the batch `x`. `embedded` is
        # of size `seq_length` * `batch_size` * `embedding_dim`
        embedded = ...

        # Flatten the embedded words and feed it to the linear layer. `flatten`
        # must be of size `batch_size` * (`seq_length` * `embedding_dim`). You
        # might need to use `permute` first.
        flatten = ...

        # Apply the linear layer and return a squeezed version
        # `l1` is of size `batch_size`
        return ...


# %%
embedding_net = EmbeddingNet(VOCAB_SIZE, EMBEDDING_DIM, SEQ_LENGTH)
print(sum(torch.numel(e) for e in embedding_net.parameters()))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

optimizer = Adam(embedding_net.parameters())
train(embedding_net, optimizer)

# %% [markdown]
# ### Training a linear classifier with a pretrained embedding
#
# Load a GloVe pretrained embedding instead

# %%
# Download GloVe word embedding
import gensim.downloader
glove_vectors = gensim.downloader.load('glove-twitter-25')

unknown_vector = glove_vectors.get_mean_vector(glove_vectors.index_to_key)
vocab_vectors = torch.tensor(np.stack([glove_vectors[e] if e in glove_vectors else unknown_vector for e in vocab.keys()]))

class GloVeEmbeddingNet(nn.Module):
    def __init__(self, seq_length, vocab_vectors, freeze=True):
        super().__init__()
        self.seq_length = seq_length

        # Define `embedding_dim` from vocabulary and the pretrained `embedding`.
        self.embedding_dim = ...
        self.embedding = ...

        self.l1 = nn.Linear(self.seq_length * self.embedding_dim, 1)

    def forward(self, x):
        # Same forward as in `EmbeddingNet`
        # `x` is of size `batch_size` * `seq_length`
        embedded = ...
        flatten = ...

        # L1 is of size batch_size
        return ...


glove_embedding_net1 = GloVeEmbeddingNet(SEQ_LENGTH, vocab_vectors, freeze=True)
print(sum(torch.numel(e) for e in glove_embedding_net1.parameters()))

optimizer = Adam(glove_embedding_net1.parameters())
train(glove_embedding_net1, optimizer)

# ## Use pretrained embedding without fine-tuning

# Define model and freeze the embedding
glove_embedding_net1 = ...

# %% [markdown]
# ### Fine-tuning the pretrained embedding

# %%
# Define model and don't freeze embedding weights
glove_embedding_net2 = ...


# %% [markdown]
# ### Recurrent neural network with frozen pretrained embedding

# %%
class RNN(nn.Module):
    def __init__(self, hidden_size, vocab_vectors, freeze=True):
        super(RNN, self).__init__()

        # Define pretrained embedding
        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)

        # Size of input `x_t` from `embedding`
        self.embedding_size = self.embedding.embedding_dim
        self.input_size = self.embedding_size

        # Size of hidden state `h_t`
        self.hidden_size = hidden_size

        # Define a GRU
        self.gru = ...

        # Linear layer on last hidden state
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, h0=None):
        # `x` is of size `seq_length` * `batch_size` and `h0` is of size 1
        # * `batch_size` * `hidden_size`

        # Define first hidden state in not provided
        if h0 is None:
            # Get batch and define `h0` which is of size 1 * # `batch_size` *
            # `hidden_size`
            batch_size = ...
            h0 = ...

        # `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim`
        embedded = self.embedding(x)

        # Define `output` and `hidden` returned by GRU:
        #
        # - `output` is of size `seq_length` * `batch_size` * `embedding_dim`
        #   and gathers all the hidden states along the sequence.
        # - `hidden` is of size 1 * `batch_size` * `embedding_dim` and is the
        #   last hidden state.
        output, hidden = ...

        # Apply a linear layer on the last hidden state to have a score tensor
        # of size 1 * `batch_size` * 1, and return a one-dimensional tensor of
        # size `batch_size`.
        return ...


rnn = RNN(hidden_size=100, vocab_vectors=vocab_vectors)
print("Number of parameters for RNN model:", sum(torch.numel(e) for e in rnn.parameters() if e.requires_grad))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, rnn.parameters()), lr=0.001)
train(rnn, optimizer)


# %% [markdown]
# ### CNN based text classification

# %%
class CNN(nn.Module):
    def __init__(self, vocab_vectors, freeze=False):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(vocab_vectors, freeze=freeze)
        self.embedding_dim = self.embedding.embedding_dim

        self.conv_0 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(3, self.embedding_dim)
        )
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(4, self.embedding_dim)
        )
        self.conv_2 = nn.Conv2d(
            in_channels=1, out_channels=100, kernel_size=(5, self.embedding_dim)
        )
        self.linear = nn.Linear(3 * 100, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input `x` is of size `seq_length` * `batch_size` and contains integers
        embedded = self.embedding(x)

        # The tensor `embedded` is of size `seq_length` * `batch_size` *
        # `embedding_dim` and should be of size `batch_size` * (`n_channels`=1)
        # * `seq_length` * `embedding_dim` for the convolutional layers. You can
        # use `transpose` and `unsqueeze` to make the transformation.

        embedded = ...

        # Tensor `embedded` is now of size `batch_size` * 1 *
        # `seq_length` * `embedding_dim` before convolution and should
        # be of size `batch_size` * (`out_channels` = 100) *
        # (`seq_length` - `kernel_size[0]` + 1) after convolution and
        # squeezing.
        # Implement the three parallel convolutions
        conved_0 = ...
        conved_1 = ...
        conved_2 = ...

        # Non-linearity step, we use ReLU activation
        conved_0_relu = ...
        conved_1_relu = ...
        conved_2_relu = ...

        # Max-pooling layer: pooling along whole sequence
        # Implement max pooling
        seq_len_0 = ...
        pooled_0 = ...
        seq_len_1 = ...
        pooled_1 = ...
        seq_len_2 = ...
        pooled_2 = ...

        # Dropout on concatenated pooled features
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))

        # Linear layer
        return self.linear(cat).squeeze()

cnn = CNN(vocab_vectors)
optimizer = optim.Adam(cnn.parameters())
train(cnn, optimizer)
