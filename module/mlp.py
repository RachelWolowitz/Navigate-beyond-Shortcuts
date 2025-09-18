import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, ETF, num_classes=10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

        self.n_feat = 100
        
        if ETF == "vanilla":
            self.classifier = nn.Linear(self.n_feat, num_classes, bias=False)
        elif ETF == "ETF":
            self.classifier = nn.Linear(self.n_feat*2, num_classes, bias=False)
            self.attr_ETF = nn.Linear(self.n_feat, num_classes, bias=False)
        elif ETF == "benign":
            self.classifier = nn.Linear(self.n_feat*2, num_classes, bias=False)

    def forward(self, x, ETF, attr_feat=None, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = self.feature(x)

        if ETF == "vanilla":
            x = self.classifier(feat)

            if return_feat:
                return x, feat
            else:
                return x
        elif ETF == "ETF":
            assert attr_feat != None
            feat = torch.cat((feat, attr_feat), dim=1)
            x = self.classifier(feat)

            if return_feat:
                return x, feat
            else:
                return x
        elif ETF == "benign":
            assert attr_feat != None
            feat = torch.cat((feat, attr_feat), dim=1)
            x = self.classifier(feat)

            if return_feat:
                return x, feat
            else:
                return x
            
