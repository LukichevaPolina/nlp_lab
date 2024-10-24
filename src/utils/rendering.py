import matplotlib.pyplot as plt

SAVE_PATH = "graphs/"

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