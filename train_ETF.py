import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import os
import scipy.linalg as scilin
from tqdm import trange, tqdm
from datetime import datetime
import numpy as np
import math
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torchvision.models as models
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from util import MultiDimAverageMeter
import time

device = 'cuda'
data_dir = "./data"
log_dir = "./log"

## [ColoredMNIST, CorruptedCIFAR10-type0, CorruptedCIFAR10-type1]
dataset = "ColoredMNIST"

if dataset == "ColoredMNIST":
    main_tag = "ColoredMNIST"
    dataset_tag = "ColoredMNIST-Skewed0.02-Severity4"
    model_tag = "MLP"
    target_attr_idx = 0
    bias_attr_idx = 1
    main_num_steps = 235 * 200
    main_valid_freq = 235
    main_batch_size = 256
    n_feat = 100
    main_optimizer_tag = "Adam"
    main_learning_rate = 1e-3
    main_weight_decay = 1e-5
elif dataset == "CorruptedCIFAR10-type0":
    main_tag = "CorruptedCIFAR10"
    dataset_tag = "CorruptedCIFAR10-Type0-Skewed0.02-Severity4"
    model_tag = "ResNet20"
    target_attr_idx = 0
    bias_attr_idx = 1
    main_num_steps = 196 * 200
    main_valid_freq = 196
    main_batch_size = 256
    n_feat = 64
    main_optimizer_tag = "SGD"
    main_learning_rate = 0.05
    main_weight_decay = 2e-4
elif dataset == "CorruptedCIFAR10-type1":
    main_tag = "CorruptedCIFAR10-1"
    dataset_tag = "CorruptedCIFAR10-Type1-Skewed0.05-Severity4"
    model_tag = "ResNet20"
    target_attr_idx = 0
    bias_attr_idx = 1
    main_num_steps = 196 * 200
    main_valid_freq = 196
    main_batch_size = 256
    n_feat = 64
    main_optimizer_tag = "Adam"
    main_learning_rate = 1e-2
    main_weight_decay = 0

print(dataset_tag)

device = torch.device(device)
start_time = datetime.now()
writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

if main_tag == "ColoredMNIST" or main_tag == "CorruptedCIFAR10":
    train_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="train",
        transform_split="train",
    )
    valid_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="eval",
        transform_split="eval",
    )
    
train_target_attr = train_dataset.attr[:, target_attr_idx]
train_bias_attr = train_dataset.attr[:, bias_attr_idx]
attr_dims = []
attr_dims.append(torch.max(train_target_attr).item() + 1)
attr_dims.append(torch.max(train_bias_attr).item() + 1)
num_classes = attr_dims[0]
        
train_dataset = IdxDataset(train_dataset)
valid_dataset = IdxDataset(valid_dataset)   

train_loader = DataLoader(
    train_dataset,
    batch_size=main_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
)

def compute_ETF(W):
    K = W.shape[0]
    W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.cpu().numpy().item()

# generate etf-classifier weight.data
def get_etf_model(feat_in, num_classes):
    a = np.random.random(size=(feat_in, num_classes))
    P, _ = np.linalg.qr(a)
    P = torch.tensor(P).float()
    I = torch.eye(num_classes)
    one = torch.ones(num_classes, num_classes)
    M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I - ((1/num_classes) * one))
    M_transposed = M.t()
    classifier = nn.Linear(feat_in, num_classes, bias=False).to(device)
    classifier.weight.data.copy_(M_transposed)
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier
      
def features_centers_similarity_std(all_labels, features):
    class_mean = []
    labels = torch.unique(all_labels)
    for label in labels:
        mean = torch.mean(features[all_labels == label], axis = 0)
        class_mean.append(mean.cpu().numpy())
        
    return class_mean

# define model and optimizer
model = get_model(model_tag, ETF="ETF", num_classes=num_classes).to(device)
model = torch.nn.DataParallel(model).to(device)
model.module.attr_ETF = get_etf_model(int(n_feat), num_classes)

for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

if main_optimizer_tag == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=main_learning_rate,
        weight_decay=main_weight_decay,
        momentum=0.9,
    )
elif main_optimizer_tag == "Adam":
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=main_learning_rate,
        weight_decay=main_weight_decay,
    )
elif main_optimizer_tag == "AdamW":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=main_learning_rate,
        weight_decay=main_weight_decay,
    )
elif main_optimizer_tag == "Adam_fixed_classifier":
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=main_learning_rate,
        weight_decay=main_weight_decay,
    )
else:
    raise NotImplementedError

### CIFAR10
'''
scheduler = lrs.StepLR(
            optimizer,
            step_size=100,
            gamma=0.8
        )
'''
scheduler = lrs.StepLR(
            optimizer,
            step_size=10,
            gamma=1.01
        )

def evaluate():
    model_path = os.path.join(log_dir, "result", main_tag, "ETF.th")
    state_dict = torch.load(model_path)['state_dict']
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     new_k = k.replace('module.', '') if 'module' in k else k
    #     new_state_dict[new_k] = v
    model.load_state_dict(state_dict)
    model.eval()

    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    z_G = 0
    z_k = []
    for index, data, attr in tqdm(valid_loader, leave=False):
        label = attr[:, target_attr_idx]
        attr_label = attr[:, bias_attr_idx]
        data = data.to(device)
        attr_label = attr_label.to(device)
        label = label.to(device)
        with torch.no_grad():
            attr_feat = torch.zeros((data.shape[0], n_feat)).to(device)
            logit, feat = model(data, ETF="ETF", attr_feat=attr_feat, return_feat=True)

            pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).long()
            z_G = torch.sum(feat, dim=0)
            z_k.append(features_centers_similarity_std(label, feat))

        attr = attr[:, [target_attr_idx, bias_attr_idx]]

        attrwise_acc_meter.add(correct.cpu(), attr.cpu())

    accs = attrwise_acc_meter.get_mean()
    valid_accs = torch.mean(accs) * 100
    eye_tsr = torch.eye(num_classes)
    valid_aligned_accs = accs[eye_tsr > 0.0].mean() * 100
    valid_skewed_accs = accs[eye_tsr == 0.0].mean() * 100

    print(f"[Eval]: Acc {valid_accs:.2f} Acc_Align {valid_aligned_accs:.2f} Acc_Skew {valid_skewed_accs:.2f}")


    z_G = 0
    z_k = dict()
    Sigma_W = 0
    for index, data, attr in tqdm(train_loader, leave=False):
        label = attr[:, target_attr_idx]
        attr_label = attr[:, bias_attr_idx]
        data = data.to(device)
        attr = attr.to(device)
        label = label.to(device)
        attr_label = attr_label.to(device)

        with torch.no_grad():
            attr_feat = torch.zeros((data.shape[0], n_feat)).to(device)
            logit, feat = model(data, ETF="ETF", attr_feat=attr_feat, return_feat=True)

            ## global
            z_G += torch.sum(feat, dim=0)
            for b in range(len(label)):
                y = label[b].item()
                if y not in z_k:
                    z_k[y] = feat[b, :]
                else:
                    z_k[y] += feat[b, :]
    
    z_G /= len(train_loader.dataset)
    CIFAR10_TRAIN_SAMPLES = 10 * [5000.]
    for i in range(len(z_k)):
        z_k[i] /= CIFAR10_TRAIN_SAMPLES[i]

    for index, data, attr in tqdm(train_loader, leave=False):
        label = attr[:, target_attr_idx]
        attr_label = attr[:, bias_attr_idx]
        data = data.to(device)
        attr = attr.to(device)
        label = label.to(device)
        attr_label = attr_label.to(device)

        with torch.no_grad():
            attr_feat = torch.zeros((data.shape[0], n_feat)).to(device)
            logit, feat = model(data, ETF="ETF", attr_feat=attr_feat, return_feat=True)

            for b in range(len(label)):
                y = label[b].item()
                Sigma_W += (feat[b, :] - z_k[y]).unsqueeze(1) @ (feat[b, :] - z_k[y]).unsqueeze(0)

    ### NC metrics
    Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)

    Sigma_B = 0
    K = len(z_k)
    for i in range(K):
        Sigma_B += (z_k[i] - z_G).unsqueeze(1) @ (z_k[i] - z_G).unsqueeze(0)
    Sigma_B /= K

    NC1 = np.trace(Sigma_W.cpu() @ scilin.pinv(Sigma_B.cpu())) / K
    print("NC1", NC1)

    w_k = model.module.classifier.weight.data
    NC2 = compute_ETF(w_k)
    NC2_1 = compute_ETF(w_k[:, :n_feat])
    NC2_2 = compute_ETF(w_k[:, n_feat:])
    print("NC2", NC2, NC2_1, NC2_2)

def train():
    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def evaluate(model, data_loader):
        model.eval()
        correct = 0
        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
        # for index, data, attr, _ in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            attr_label = attr[:, bias_attr_idx]
            data = data.to(device)
            attr_label = attr_label.to(device)
            label = label.to(device)
            with torch.no_grad():
                attr_feat = torch.zeros((data.shape[0], n_feat)).to(device)
                
                color_etf = []
                for a_l in attr_label:
                    color_etf.append(model.module.attr_ETF.weight.data[a_l])
                color_etf = torch.stack(color_etf)
                
                logit, feat = model(data, ETF="ETF", attr_feat=attr_feat, return_feat=True)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct += (pred == label).long().sum()

        accs = correct*1.0 / len(data_loader.dataset) * 100.
        print(accs)
        model.train()

    bar = trange(main_num_steps)
    l_acc = []
    l_clean = []
    model.train()
    
    for step in bar:
        if main_tag == "bffhq" or main_tag == "bar" or main_tag == "dogs_and_cats":
            try:
                index, data, attr, _ = next(train_iter)
            except:
                train_iter = iter(train_loader)
                index, data, attr, _ = next(train_iter)
        else:
            try:
                index, data, attr = next(train_iter)
            except:
                train_iter = iter(train_loader)
                index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, target_attr_idx]
        attr_label = attr[:, bias_attr_idx]

        color_etf = []
        for a_l in attr_label:
            color_etf.append(model.module.attr_ETF.weight.data[a_l])
        color_etf = torch.stack(color_etf)

        logit, feat = model(data, ETF="ETF", attr_feat=color_etf, return_feat=True)
        loss_per_sample_acc = label_criterion(logit, label)
        loss_acc = loss_per_sample_acc.mean()

        attr_feat = torch.zeros((data.shape[0], n_feat)).to(device)
        logit_clean, feat_clean = model(data, ETF="ETF", attr_feat=attr_feat, return_feat=True)
        logit_clean_minus = logit - logit_clean
        loss_per_sample_clean = label_criterion(logit_clean_minus, attr_label)
        loss_clean = loss_per_sample_clean.mean()
        
        loss = loss_acc + 0.8 * loss_clean
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        l_acc.append(loss_acc.item())
        l_clean.append(loss_clean.item())
        bar.set_postfix({'loss_acc': '{:.2f}'.format(np.mean(l_acc)), 'loss_clean': '{:.2f}'.format(np.mean(l_clean))})

        if step != 0 and step % main_valid_freq == 0:
            scheduler.step()
            evaluate(model, valid_loader)

            
    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    model_path = os.path.join(log_dir, "result", main_tag, "ETF.th")
    state_dict = {
        'steps': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)

# train()
evaluate()