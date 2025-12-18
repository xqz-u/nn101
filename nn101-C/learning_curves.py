import matplotlib.pyplot as plt
import pandas as pd


def plot_training(
    train_logs_path: str, batch_train_logs_path: str, test_logs_path: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Main Overview (Loss per Epoch) ---
    df_train = pd.read_csv(train_logs_path)
    df_test = pd.read_csv(test_logs_path)

    df_epoch = pd.merge(df_train, df_test, on="epoch", suffixes=("_train", "_test"))

    color = "tab:red"
    ax1.set_xlabel("Epoch")  # TODO integer
    ax1.set_ylabel("Avg Train Loss", color=color)
    ax1.plot(
        df_epoch["epoch"],
        df_epoch["loss"],
        color=color,
        marker="o",
        label="Train Loss",
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax1_right = ax1.twinx()
    color = "tab:blue"
    ax1_right.set_ylabel("Test Accuracy", color=color)
    ax1_right.plot(
        df_epoch["epoch"],
        df_epoch["accuracy"],
        color=color,
        marker="x",
        linestyle="--",
        label="Test Accuracy",
    )
    ax1_right.tick_params(axis="y", labelcolor=color)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    ax1.set_title("Training Overview (Per Epoch)")

    # --- Plot 2: Dynamics (Loss per Batch) ---
    # This helps you see if the loss is oscillating within an epoch
    df_batch = pd.read_csv(batch_train_logs_path)

    ax2.plot(df_batch["batch"], df_batch["loss"], alpha=0.5, label="Batch Loss")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Batch Loss")
    ax2.set_title("Training Dynamics (Per Batch)")
    ax2.legend()

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    fig = plot_training("logs/train.csv", "logs/train_batch.csv", "logs/test.csv")
    fig.show()
