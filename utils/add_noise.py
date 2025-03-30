import torch


def add_noise(model):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()) * 0.05) #↑
            

@torch.no_grad()
def add_noise_to_weights(m):
    if hasattr(m, 'weight'):
        m.weight.add_(torch.randn(m.weight.size()) * 0.1)