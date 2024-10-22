import torch
from torch import optim
from src.data.combined_datamodule import CombinedDatamodule
from src.models.cnn.conv_psycho_net import ConvPsychoNet
import random
import numpy as np

from torch import nn

from torcheval.metrics.classification.accuracy import MulticlassAccuracy
from torcheval.metrics.classification.f1_score import MulticlassF1Score


SEED = 4200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(chekpoint_save, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32) -> None: # Need to return metrics and losses from each epoch and mean
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datamodule = CombinedDatamodule(X_train, y_train, X_test, y_test, train_bs=batch_size, test_bs=4)
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    psycho_net = ConvPsychoNet(in_channels=1, out_channels=7).to(device)
    print("[INFO] Start PsychoNet initialization")
    psycho_net.apply(ConvPsychoNet.initialize)

    optimizer = optim.AdamW(psycho_net.parameters(), lr=2e-3, betas=(0.5, 0.9), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97048695)
    ce_loss = nn.CrossEntropyLoss(torch.Tensor([4.0, 5.0, 1.0, 1.0, 16.0, 7.5, 1.2])).to(device)
    
    accuracy = MulticlassAccuracy(None, num_classes=7).to(device)
    f1_score = MulticlassF1Score("micro", num_classes=7).to(device)

    best_f1score = 0.0
    for epoch in range(num_epochs):
        psycho_net.train()
        accuracy.reset()
        f1_score.reset()

        celosses = []
        metrics = {"accuracy": [], "f1score": []}
        train_losses = {"ce": []}
        val_losses = {"ce": []}
        for X, y in train_dataloader:
            psycho_net.zero_grad()
            pred = psycho_net(X)

            celoss = ce_loss(pred, y)
            celosses.append(celoss.item())

            celoss.backward()
            optimizer.step()

            accuracy.update(pred, y)
            f1_score.update(pred, y)
            
        metrics["accuracy"].append(accuracy.compute())
        metrics["f1score"].append(f1_score.compute())
        train_losses["ce"].append(np.array(celosses).mean())

        val_loss = val(psycho_net, device, test_dataloader)
        val_losses["ce"].append(val_loss)

        if epoch % 5 == 0:
            print(f"EPOCH={epoch}/TRAIN")
            print(f"train_ce_loss={train_losses["ce"][-1]}, val_ce_loss={val_losses['ce'][-1]}, f1score={metrics['f1score'][-1]}")

        if metrics["f1score"][-1] > best_f1score:
            best_f1score = metrics["f1score"][-1]
            torch.save(psycho_net.state_dict(), f"checkpoints/{chekpoint_save}-epoch{epoch}.pt")

        scheduler.step()
    
def val(psycho_net, device, test_dataloader):
    psycho_net.eval()

    celosses = []
    ce_loss = nn.CrossEntropyLoss(torch.Tensor([4.0, 5.0, 1.0, 1.0, 16.0, 7.5, 1.2])).to(device)

    with torch.no_grad():
        for X, y in test_dataloader:
            pred = psycho_net(X)

            celoss = ce_loss(pred, y)
            celosses.append(celoss.item())

    psycho_net.train()
    return np.array(celosses).mean()
    

