import torch
from src.models.cnn.conv_psycho_net import ConvPsychoNet

def cnn_infer(checkpoint_path, X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    psycho_net = ConvPsychoNet(1, 7).to(device)
    psycho_net.load_state_dict(torch.load(checkpoint_path))
    psycho_net.eval()

    with torch.no_grad():
        X = torch.from_numpy(X.toarray().reshape(-1, 14951)).float().unsqueeze(dim=0).unsqueeze(dim=1)
        pred = psycho_net(X)

        y_pred = torch.softmax(pred, dim=1).armax(dim=1)

    # TODO: probably necessary convert to numpy array
    return y_pred
