"""Week 1 EDA: three required visualisations (distributions, correlation, balance)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C
from .data_pipeline import load_and_clean


def run():
    C.ensure_dirs()
    df = load_and_clean(verbose=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, C.NUMERIC_COLS):
        sns.histplot(data=df, x=col, hue=C.TARGET_COL, kde=True, ax=ax)
        ax.set_title(f"{col} by Reaction_Score")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_distributions.png", dpi=150); plt.close(fig)

    num = df[C.NUMERIC_COLS + [C.COUNT_COL, C.TARGET_COL]]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_correlation.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.countplot(data=df, x=C.TARGET_COL, ax=ax)
    ax.set_title("Class balance (Reaction_Score)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_class_balance.png", dpi=150); plt.close(fig)
    print("EDA figures saved to", C.FIG_DIR)


if __name__ == "__main__":
    run()
