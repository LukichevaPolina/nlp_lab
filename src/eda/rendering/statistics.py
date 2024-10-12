import matplotlib.pyplot as plt
import pandas as pd

SAVE_PATH = "../../../graphs/"

def plot_class_distribution(data: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9, 5))
    plt.hist(data["status"], bins=50, histtype="barstacked")
    fig.suptitle("Class distribution", fontsize=20)
    plt.xlabel("status", fontsize=18)
    plt.ylabel("frequently", fontsize=16)
    fig.savefig(f"{SAVE_PATH}class_distribution.png")