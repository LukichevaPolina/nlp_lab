import torch
from src.models.cnn.conv_psycho_net import ConvPsychoNet


def cnn_infer(checkpoint_path, X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    psycho_net = ConvPsychoNet(1, 7).to(device)
    psycho_net.load_state_dict(torch.load(checkpoint_path))
    psycho_net.eval()

    with torch.no_grad():
        shape = X.toarray().shape
        X = torch.from_numpy(X.toarray()).float().unsqueeze(
            dim=1).unsqueeze(dim=2).reshape(shape[0], 1, -1, 599)

        pred = psycho_net(X).squeeze()

        y_pred = torch.softmax(pred, dim=1).argmax(dim=1)

    return y_pred
