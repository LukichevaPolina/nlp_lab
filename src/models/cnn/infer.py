import torch
from src.models.cnn.conv_psycho_net import ConvPsychoNet

def cnn_infer(checkpoint_path, X):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    psycho_net = ConvPsychoNet(1, 7).to(device)
    psycho_net.load_state_dict(checkpoint)
    psycho_net.eval()

    with torch.no_grad():
        pred = psycho_net(X)

        y_pred = torch.softmax(pred, dim=1).armax(dim=1)

    # TODO: probably necessary convert to numpy array
    return y_pred
