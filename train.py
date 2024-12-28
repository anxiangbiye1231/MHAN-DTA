# %%
import os
import pandas as pd
import torch
import warnings
import numpy as np
import random
from dataset import Diadataset, collate_fn
from torch.utils.data import DataLoader
from model_sensitivity import MHAN
from tqdm import tqdm
import metrics
from config.config_dict import Config
from log.train_logger import TrainLogger
# %%

def train(train_dataloader, criterion, device, model, optimizer):
    model.train()
    tbar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100)
    losses = []
    for i, c_data in enumerate(tbar):
        seq_trans_data = c_data[0].to(device)
        pocket_res_data = c_data[1].to(device)
        seq_info_data = c_data[2].to(device)
        ligand_data = c_data[3].to(device)
        smiles_info_data = c_data[4].to(device)
        affinity = c_data[3].y
        y_pred = model(seq_trans_data, pocket_res_data, seq_info_data, ligand_data, smiles_info_data).squeeze()
        y_true = affinity.float().squeeze()

        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred, y_true).cuda()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    m_losses = np.mean(losses)

    return m_losses

def valid(model, valid_dataloader, criterion, device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    tbar = tqdm(valid_dataloader, total=len(valid_dataloader), ncols=80)
    for i, c_data in enumerate(tbar):

        seq_trans_data = c_data[0].to(device)
        pocket_res_data = c_data[1].to(device)
        seq_info_data = c_data[2].to(device)
        ligand_data = c_data[3].to(device)
        smiles_info_data = c_data[4].to(device)
        affinity = c_data[3].y
        with torch.no_grad():
            y_pred = model(seq_trans_data, pocket_res_data, seq_info_data, ligand_data, smiles_info_data).squeeze()
        y_true = affinity.float().squeeze()

        assert y_pred.shape == y_true.shape
        loss = criterion(y_pred, y_true).cuda()
        losses.append(loss.item())
        outputs.append(y_pred.cpu().detach().numpy().reshape(-1))
        targets.append(y_true.cpu().detach().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    evaluation = {
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }
    ml = np.mean(losses)

    return ml, evaluation

def labels_random(labels_dict):
    data_list = list(labels_dict.items())
    random.shuffle(data_list)
    shuffled_data_dict = dict(data_list)
    return shuffled_data_dict

def main(config):
    args = config.get_config()
    datasets_path = args.get('data_root')
    batch_size = args.get("batch_size")
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    train_path = datasets_path + 'train'
    valid_path = datasets_path + 'test'
    train_labels = pd.read_csv(datasets_path + 'train.csv')
    valid_labels = pd.read_csv(datasets_path + 'test.csv')

    prot_len = 1400
    pock_dis = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_labels_dict = {id[0]: id[1] for id in train_labels.values}
    valid_labels_dict = {id[0]: id[1] for id in valid_labels.values}
    train_labels_s = labels_random(train_labels_dict)
    valid_labels_s = labels_random(valid_labels_dict)
    train_dataset_complex = Diadataset(train_path, train_labels_s, prot_len=prot_len, pock_dis=pock_dis)
    valid_dataset_complex = Diadataset(valid_path, valid_labels_s, prot_len=prot_len, pock_dis=pock_dis)
    train_dataloader = DataLoader(train_dataset_complex, batch_size=batch_size, shuffle=True, num_workers=8,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset_complex, batch_size=batch_size, shuffle=False, num_workers=8,
                                  collate_fn=collate_fn)
    for repeat in range(repeats):
        args['repeat'] = repeat
        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_dataset_complex)}")
        logger.info(f"valid data: {len(valid_dataset_complex)}")

        model = MHAN()
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 1.2e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
        criterion = torch.nn.MSELoss()

        best_rmse = float('inf')
        for epoch in range(epochs):
            ll = train(train_dataloader, criterion, device, model, optimizer)
            l, evaluation = valid(model, valid_dataloader, criterion, device)
            msg = "epoch-%d, train_loss-%.3f, valid_loss-%.3f, c_index-%.3f, RMSE-%.3f, MAE-%.3f, SD-%.3f, CORR-%.3f" \
                  % (epoch,
                     ll,
                     l,
                     evaluation['c_index'],
                     evaluation['RMSE'],
                     evaluation['MAE'],
                     evaluation['SD'],
                     evaluation['CORR'])
            logger.info(msg)
            model_path = os.path.join(logger.get_model_dir(), f'{msg}.pth')
            if (evaluation['RMSE'] < best_rmse):
                best_rmse, best_mae = evaluation['RMSE'], evaluation['MAE']
                torch.save({'model': model.state_dict()}, model_path)
        scheduler.step()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cfg = 'TrainConfig_m'
    config = Config(cfg)
    main(config)
