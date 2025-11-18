# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from pathlib import Path
import nn_model


class CNN_1D_2L(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(2,stride=2)
        )
        
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.AvgPool1d(2,stride=2)
        )
        
        self.linear1 = nn.Linear(self.n_in*128 //4, 4)

        
    def forward(self, x):
        x = x.view(-1, 1, self.n_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.n_in*128//4)
        return self.linear1(x)


# Functions for training
def get_dataloader(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def loss_batch(model, loss_func, xb, yb, opt=None):
    out = model(xb)
    loss = loss_func(out, yb)
    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False):
    print(
        'EPOCH', '\t', 
        'Train Loss', '\t',
        'Val Loss', '\t', 
        'Train Acc', '\t',
        'Val Acc', '\t')
    # Initialize dic to store metrics for each epoch.
    metrics_dic = {}
    metrics_dic['train_loss'] = []
    metrics_dic['train_accuracy'] = []
    metrics_dic['val_loss'] = []
    metrics_dic['val_accuracy'] = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_examples = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, batch_size, pred = loss_batch(model, loss_func, xb, yb, opt)
            if train_metric == False:
                train_loss += loss*batch_size
                num_examples += batch_size

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, _ = validate(model, valid_dl, loss_func)
            if train_metric:
                train_loss, train_accuracy, _ = validate(model, train_dl, loss_func)
            else:
                train_loss = train_loss / num_examples

        metrics_dic['val_loss'].append(val_loss)
        metrics_dic['val_accuracy'].append(val_accuracy)
        metrics_dic['train_loss'].append(train_loss)
        metrics_dic['train_accuracy'].append(train_accuracy)
        
        print(
            f'{epoch} \t', 
            f'{train_loss:.05f}', '\t',
            f'{val_loss:.05f}', '\t', 
            f'{train_accuracy:.05f}', '\t'
            f'{val_accuracy:.05f}', '\t')
        
    metrics = pd.DataFrame.from_dict(metrics_dic)

    return model, metrics

def validate(model, dl, loss_func):
    total_loss = 0.0
    total_size = 0
    predictions = []
    y_true = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for xb, yb in dl: 
        xb, yb = xb.to(device), yb.to(device)
        loss, batch_size, pred = loss_batch(model, loss_func, xb, yb)
        total_loss += loss*batch_size
        total_size += batch_size
        predictions.append(pred)
        y_true.append(yb.cpu().numpy())
    mean_loss = total_loss / total_size
    predictions = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    accuracy = np.mean((predictions == y_true))
    return mean_loss, accuracy, (y_true, predictions)

if __name__ == "__main__":
    working_dir = Path('.')
    DATA_PATH = Path("./Data")
    save_model_path = working_dir / 'Model'
    DE_path = DATA_PATH / '48k_DE'
    FE_path = DATA_PATH / '12k_FE'
    
    for path in [DATA_PATH, save_model_path]:
        if not path.exists():
            path.mkdir(parents=True)
            
    bs = 32
    lr = 0.001
    wd = 1e-5
    betas=(0.99, 0.999)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random_seed = 42
    
    df1 = pd.read_csv("./PreData/48de_signals.csv")
    df2 = pd.read_csv("./PreData/12de_signals.csv")
    df3 = pd.read_csv("./PreData/fe_signals.csv")
    df_all = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    
    features = df_all.columns[2:]
    target = 'label'

    X_train, X_valid, y_train, y_valid = train_test_split(df_all[features], 
                                                          df_all[target], 
                                                          test_size=0.20, random_state=random_seed, shuffle=True
                                                         )
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    X_valid = torch.tensor(X_valid.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_valid = torch.tensor(y_valid.values, dtype=torch.long)
    
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    train_dl, valid_dl = get_dataloader(train_ds, valid_ds, bs)

    model = nn_model.CNN_1D_2L(len(features))
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    loss_func = CrossEntropyLoss()
    epochs = 35
    model, metrics = fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=True)
    
    
    torch.save(model.state_dict(), save_model_path / 'model.pth')
    model2 = nn_model.CNN_1D_2L(len(features))
    loss_func = CrossEntropyLoss()
    
    