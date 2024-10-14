from collections import Counter 
import matplotlib.pyplot as plt
import logging as log
import pandas as pd

SAVE_PATH = "../../../graphs/"

def plot_data_information(data: pd.DataFrame) -> None:
    log.info("PLOT: save data information into graphs")
    sentences = []
    data["statement"].apply(lambda x: sentences.append(" ".join(x)))
    total_words = " ".join(sentences).split()
    word_frequency = Counter(total_words).most_common(20)
    words = [word for word, _ in word_frequency]
    counts = [counts for _, counts in word_frequency]

    fig = plt.figure(figsize=(15, 5))
    plt.bar(words, counts)
    plt.title("20 most frequent words", fontsize=15)
    plt.ylabel("frequency", fontsize=9)
    plt.xlabel("words", fontsize=9)
    fig.savefig(f"{SAVE_PATH}data_info.png")