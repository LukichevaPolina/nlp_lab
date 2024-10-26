import torch
from src.models.linear.linear_psycho_net import LinPsychoNet


def linear_infer(checkpoint_path, X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpsycho_net = LinPsychoNet(1, 7).to(device)
    lpsycho_net.load_state_dict(torch.load(checkpoint_path))
    lpsycho_net.eval()

    with torch.no_grad():
        shape = X.toarray().shape
        X = torch.from_numpy(X.toarray()).float().unsqueeze(
            dim=1).unsqueeze(dim=2).reshape(shape[0], 1, -1, 599)

        pred = lpsycho_net(X).squeeze()

        y_pred = torch.softmax(pred, dim=1).argmax(dim=1)

    return y_pred
