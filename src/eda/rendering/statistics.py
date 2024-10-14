import matplotlib.pyplot as plt
import pandas as pd
import logging as log

SAVE_PATH = "../../../graphs/"

def plot_class_distribution(data: pd.DataFrame) -> None:
    log.info("PLOT: save class distribution into graphs/")
    fig = plt.figure(figsize=(9, 5))
    plt.hist(data["status"], bins=50, histtype="barstacked")
    fig.suptitle("Class distribution", fontsize=15)
    plt.xlabel("status", fontsize=9)
    plt.ylabel("frequency", fontsize=9)
    fig.savefig(f"{SAVE_PATH}class_distribution.png")

def class_features_distribution(data, plot_title, function, function_name) -> None:
    log.info(f"PLOT: save statistics {plot_title} graphs into graphs/")
    classes = data["status"].unique()
    fig, ax = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=True, figsize=(17, 10))
    fig.suptitle(plot_title, fontsize=15)

    for i, class_name in enumerate(classes):
        subdata = data[data["status"] == class_name]
        features = subdata["statement"].apply(function)
        ax[i // 4, i % 4].hist(features, bins=25, color="g")
        ax[i // 4, i % 4].title.set_text(f"{class_name}")
        ax[i // 4, i % 4].set(xlabel=f"{function_name}", ylabel="frequency")
    
    fig.savefig(f"{SAVE_PATH}class_features_distribution_{function_name}")