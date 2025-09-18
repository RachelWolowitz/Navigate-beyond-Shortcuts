import torch.nn as nn
from module.resnet import resnet20, resnet18, Pre_train_ResNet18
from module.mlp import MLP
from torchvision.models import resnet50

def get_model(model_tag, ETF, num_classes):
    if model_tag == "ResNet20":
        return resnet20(num_classes=num_classes, ETF=ETF)
    elif model_tag == "ResNet18":
        return resnet18(num_classes=num_classes, ETF=ETF)
    elif model_tag == "ResNet50":
        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model
    elif model_tag == "MLP":
        return MLP(ETF=ETF, num_classes=num_classes)
    else:
        raise NotImplementedError
