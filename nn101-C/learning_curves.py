import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_training(
    train_logs_path: str, batch_train_logs_path: str, test_logs_path: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Main Overview (Loss per Epoch) ---
    df_train = pd.read_csv(train_logs_path)
    df_test = pd.read_csv(test_logs_path)

    df_epoch = pd.merge(df_train, df_test, on="epoch", suffixes=("_train", "_test"))
    df_epoch["epoch"] += 1  # Start counting epochs at 1

    color = "tab:red"
    ax1.plot(
        df_epoch["epoch"],
        df_epoch["loss"],
        color=color,
        marker="o",
        label="Train Loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(df_epoch["epoch"])
    ax1.set_ylabel("Train Loss")
    ax1.set_yticks(np.linspace(df_epoch["loss"].min(), df_epoch["loss"].max(), 7))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax1_right = ax1.twinx()
    color = "tab:blue"
    ax1_right.plot(
        df_epoch["epoch"],
        df_epoch["accuracy"],
        color=color,
        marker="x",
        linestyle="--",
        label="Test Accuracy",
    )
    ax1_right.set_ylabel("Test Accuracy")
    ax1_right.set_yticks(
        np.linspace(df_epoch["accuracy"].min(), df_epoch["accuracy"].max(), 7)
    )
    ax1_right.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    ax1.set_title("Training Overview (Per Epoch)")

    # --- Plot 2: Dynamics (Loss per Batch) ---
    # This helps you see if the loss is oscillating within an epoch
    df_batch = pd.read_csv(batch_train_logs_path)

    ax2.plot(df_batch["batch"], df_batch["loss"], label="Batch Loss")
    ax2.set_xlabel("Batch")
    ax2.set_xticks(
        df_batch.loc[::30, "batch"].tolist()[:-1] + [df_batch["batch"].max().tolist()]
    )
    ax2.set_ylabel("Batch Loss")
    ax2.set_yticks(np.linspace(df_batch["loss"].min(), df_batch["loss"].max(), 7))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.set_title("Training Dynamics (Per Batch)")
    ax2.legend()

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    train_logs_path = "logs/train.csv"
    batch_train_logs_path = "logs/train_batch.csv"
    test_logs_path = "logs/test.csv"

    fig = plot_training(train_logs_path, batch_train_logs_path, test_logs_path)
    fig.show()

    savepath = "learning_curves.svg"

    if savepath is not None:
        fig.savefig(savepath, format="svg")
        print(f"Saved plot to '{savepath}'")
