import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# used for parsing model names and vectorizer types from filenames with matrics
MODEL_IDX = 3
VECTORIZER_IDX = 4

SAVE_PATH = "graphs/"
METRICS_PATH = "metrics/"


def plot_learning_curve(train_loss, val_loss, name):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Learning curve", fontsize=15)

    ax.plot(train_loss, 'b', label='train')
    ax.plot(val_loss, 'y', label='val')
    ax.legend()

    plt.xlabel('epochs', fontsize=9)
    plt.ylabel('loss', fontsize=9)

    fig.savefig(f"{SAVE_PATH}{name}.png")


def plot_accuracy_curve(train_accuracy, val_accuracy, name):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Accuracy curve", fontsize=15)

    ax.plot(train_accuracy, 'b', label='train')
    ax.plot(val_accuracy, 'y', label='val')
    ax.legend()

    plt.xlabel('epochs', fontsize=9)
    plt.ylabel('accuracy', fontsize=9)

    fig.savefig(f"{SAVE_PATH}{name}.png")


def plot_f1_curve(train_f1, val_f1, name):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("f1 curve", fontsize=15)

    ax.plot(train_f1, 'b', label='train')
    ax.plot(val_f1, 'y', label='val')
    ax.legend()

    plt.xlabel('epochs', fontsize=9)
    plt.ylabel('f1', fontsize=9)

    fig.savefig(f"{SAVE_PATH}{name}.png")


def plot_metrics(files=None, plot_name="plot_val_metrics_all"):
    extention = ".json"
    l = [_ for _ in os.listdir(METRICS_PATH)]
    if files == None:
        files = [_ for _ in os.listdir(METRICS_PATH) if _.endswith(extention)]
    n_models = len(files)

    results = {"models": [], "f1_weighted": [], "accuracy": []}
    for i, filename in enumerate(files):
        result_params = filename.split("_")
        model_name = result_params[MODEL_IDX] + \
            "_" + result_params[VECTORIZER_IDX]
        results["models"].append(model_name)
        with open(METRICS_PATH + filename, 'r') as file:
            model_res = json.load(file)
            results["f1_weighted"].append(model_res["f1_weighted"])
            results["accuracy"].append(model_res["accuracy"])

    barWidth = 1 / (n_models + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Metrics results on test data", fontsize=15)

    coordinates = [np.arange(n_models)]
    for i in range(1):
        coordinates.append([x + barWidth for x in coordinates[i]])

    ax.bar(coordinates[0], results['f1_weighted'],
           width=barWidth, label='f1_weighted')
    ax.bar(coordinates[1], results['accuracy'],
           width=barWidth, label='accuracy')

    plt.xlabel('Model', fontweight='bold', fontsize=15)
    plt.ylabel('Metric value', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth/2 for r in range(n_models)], results['models'])

    plt.legend()

    fig.savefig(f"{SAVE_PATH}{plot_name}.png")
