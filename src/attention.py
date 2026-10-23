
import numpy as np
import matplotlib.pyplot as plt


BASES = ["A", "C", "G", "U"]


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode an RNA sequence as one-hot vectors."""

    sequence = sequence.strip().upper()

    if not sequence:
        raise ValueError("Please enter an RNA sequence.")

    invalid = set(sequence) - set(BASES)
    if invalid:
        raise ValueError(
            f"Invalid RNA bases: {', '.join(sorted(invalid))}. "
            "Use only A, C, G, and U."
        )

    mapping = {base: index for index, base in enumerate(BASES)}

    encoded = np.zeros((len(sequence), len(BASES)), dtype=float)

    for position, base in enumerate(sequence):
        encoded[position, mapping[base]] = 1.0

    return encoded


def self_attention(encoded: np.ndarray):
    """Demonstrate scaled dot-product self-attention."""

    if encoded.ndim != 2 or encoded.shape[0] == 0:
        raise ValueError("Input must be a non-empty 2D array.")

    dimension = encoded.shape[1]

    scores = encoded @ encoded.T / np.sqrt(dimension)

    # Numerically stable softmax.
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    output = weights @ encoded

    return weights, output


def plot_attention(weights: np.ndarray, sequence: str):
    """Plot an attention heatmap for the input sequence."""

    sequence = sequence.strip().upper()

    if weights.shape != (len(sequence), len(sequence)):
        raise ValueError("Attention matrix dimensions do not match sequence.")

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(weights, cmap="viridis", aspect="auto")

    ax.set_title("Toy RNA Self-Attention Heatmap")
    ax.set_xlabel("Sequence position")
    ax.set_ylabel("Sequence position")

    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))
    ax.set_yticks(range(len(sequence)))
    ax.set_yticklabels(list(sequence))

    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()

    return fig
