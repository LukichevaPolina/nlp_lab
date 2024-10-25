import torch
from src.models.linear.linear_psycho_net import LinPsychoNet

def linear_infer(checkpoint_path, X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpsycho_net = LinPsychoNet(1, 7).to(device)
    lpsycho_net.load_state_dict(torch.load(checkpoint_path))
    lpsycho_net.eval()

    with torch.no_grad():
        X = torch.from_numpy(X.toarray().reshape(-1, 599)).float().unsqueeze(dim=0).unsqueeze(dim=1)
        pred = lpsycho_net(X)

        y_pred = torch.softmax(pred, dim=1).armax(dim=1)

    # TODO: probably necessary convert to numpy array
    return y_pred
