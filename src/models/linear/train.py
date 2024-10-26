import torch
from torch import optim
from src.data.combined_datamodule import CombinedDatamodule
from src.models.linear.linear_psycho_net import LinPsychoNet
import random
import numpy as np

from torch import nn

from torcheval.metrics.classification.accuracy import MulticlassAccuracy
from torcheval.metrics.classification.f1_score import MulticlassF1Score


SEED = 4200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def linear_train(chekpoint_save, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    datamodule = CombinedDatamodule(X_train, y_train, X_test, y_test, train_bs=batch_size, test_bs=batch_size)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    lpsycho_net = LinPsychoNet(in_channels=1, out_channels=7).to(device)
    print("[INFO] Start PsychoNet initialization")
    lpsycho_net.apply(LinPsychoNet.initialize)

    optimizer = optim.AdamW(lpsycho_net.parameters(), lr=1e-3, betas=(0.7, 0.9), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97048695)
    ce_loss = nn.CrossEntropyLoss(torch.Tensor([4.0, 5.0, 1.0, 1.0, 16.0, 8, 1.5])).to(device)
    
    accuracy = MulticlassAccuracy(average="macro", num_classes=7).to(device)
    f1_score = MulticlassF1Score(average="micro", num_classes=7).to(device)

    best_f1score = 0.0
    train_metrics = {"accuracy": [], "f1score": []}
    val_metrics = {"accuracy": [], "f1score": []}
    train_losses = {"ce": []}
    val_losses = {"ce": []}
    print(f"[INFO] Start training, epochs = {num_epochs}")
    for epoch in range(num_epochs):
        lpsycho_net.train()
        accuracy.reset()
        f1_score.reset()

        celosses = []
        for X, y in train_dataloader:
            X = X.unsqueeze(dim=1)
            lpsycho_net.zero_grad()
            pred = lpsycho_net(X).squeeze()
            celoss = ce_loss(pred, y)
            celosses.append(celoss.item())

            celoss.backward()
            optimizer.step()
            
            y_pred = torch.softmax(pred, dim=1).argmax(dim=1)
            accuracy.update(y_pred, y)
            f1_score.update(y_pred, y)
            
        train_metrics["accuracy"].append(accuracy.compute().item())
        train_metrics["f1score"].append(f1_score.compute().item())
        train_losses["ce"].append(np.array(celosses).mean())

        val_loss, val_accuracy, val_f1 = linear_val(lpsycho_net, device, test_dataloader)
        val_metrics["accuracy"].append(val_accuracy)
        val_metrics["f1score"].append(val_f1)
        val_losses["ce"].append(val_loss)

        if epoch % 5 == 0:
            print(f"EPOCH={epoch}/TRAIN")
            print(f"train_ce_loss={train_losses["ce"][-1]}, val_ce_loss={val_losses['ce'][-1]}, train_f1score={train_metrics['f1score'][-1]}, val_f1score={val_metrics["f1score"][-1]}")

        scheduler.step()

        if val_f1 > best_f1score:
            best_f1score = val_f1
            torch.save(lpsycho_net.state_dict(), f"{chekpoint_save}")

    print(f"[INFO] Training end")
    return train_metrics, val_metrics, train_losses, val_losses
    
def linear_val(lpsycho_net, device, test_dataloader):
    lpsycho_net.eval()

    celosses = []
    ce_loss = nn.CrossEntropyLoss(torch.Tensor([4.0, 5.0, 1.0, 1.0, 16.0, 8, 1.5])).to(device)
    f1_score = MulticlassF1Score(average="micro", num_classes=7).to(device)
    accuracy = MulticlassAccuracy(average="macro", num_classes=7).to(device)
    f1_score.reset()
    accuracy.reset()
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.unsqueeze(dim=1)
            pred = lpsycho_net(X).squeeze()

            celoss = ce_loss(pred, y)
            celosses.append(celoss.item())

            y_pred = torch.softmax(pred, dim=1).argmax(dim=1)
            f1_score.update(y_pred, y)
            accuracy.update(y_pred, y)

    lpsycho_net.train()
    return np.array(celosses).mean(), accuracy.compute().item(), f1_score.compute().item()
    

