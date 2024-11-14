import os
import pandas as pd
import torch
import warnings
import numpy as np
from dataset import Diadataset, collate_fn
from torch.utils.data import DataLoader
from model import MHAN
from tqdm import tqdm
from config.config_dict import Config
from log.train_logger import TrainLogger
import metrics
def test(model, loader_compound, criterion, device):
    model.eval()
    losses = []
    outputs = []
    targets = []
    tbar = tqdm(loader_compound, total=len(loader_compound), ncols=80)
    for i, c_data in enumerate(tbar):

        seq_trans_data = c_data[0].to(device)
        pocket_res_data = c_data[1].to(device)
        seq_info_data = c_data[2].to(device)
        ligand_data = c_data[3].to(device)
        smiles_info_data = c_data[4].to(device)
        affinity = c_data[3].y
        with torch.no_grad():
            y_pred = model(seq_trans_data, pocket_res_data, seq_info_data, ligand_data, smiles_info_data)
            y_pred = y_pred.squeeze()
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

    return evaluation

def main(config):
    args = config.get_config()
    datasets_path = args.get('data_root')
    batch_size = args.get("batch_size")

    internal_test = datasets_path + 'internal_test'
    test_2013 = datasets_path + 'test_2013'
    test_2016 = datasets_path + 'test_2016'
    test_hiq = datasets_path + 'test_hiq'

    labels_test = pd.read_csv(datasets_path + f'test.csv')
    labels_2013 = pd.read_csv(datasets_path + f'test_2013.csv')
    labels_2016 = pd.read_csv(datasets_path + f'test_2016.csv')
    labels_hiq = pd.read_csv(datasets_path + f'test_hiq.csv')

    prot_len = 1400
    pock_dis = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels_dict = {id[0]: id[1] for id in labels_test.values}
    labels2013_dict = {id[0]: id[1] for id in labels_2013.values}
    labels2016_dict = {id[0]: id[1] for id in labels_2016.values}
    labels_hiq_dict = {id[0]: id[1] for id in labels_hiq.values}

    dataset_complex = Diadataset(internal_test, labels_dict, prot_len=prot_len, pock_dis=pock_dis)
    dataset2013_complex = Diadataset(test_2013, labels2013_dict, prot_len=prot_len, pock_dis=pock_dis)
    dataset2016_complex = Diadataset(test_2016, labels2016_dict, prot_len=prot_len, pock_dis=pock_dis)
    dataset_hiq_complex = Diadataset(test_hiq, labels_hiq_dict, prot_len=prot_len, pock_dis=pock_dis)

    dataloader = DataLoader(dataset_complex, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    dataloader2013 = DataLoader(dataset2013_complex, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    dataloader2016 = DataLoader(dataset2016_complex, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    dataloader_hiq = DataLoader(dataset_hiq_complex, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = MHAN()
    model = model.to(device)
    # save_dir
    model_list = [
        '/media/ST-18T/ljr/DTA/MHAN-DTA/models/model_0.pth',
        '/media/ST-18T/ljr/DTA/MHAN-DTA/models/model_2.pth',
        '/media/ST-18T/ljr/DTA/MHAN-DTA/models/model_1.pth',
        '/media/ST-18T/ljr/DTA/MHAN-DTA/models/model_3.pth',
        '/media/ST-18T/ljr/DTA/MHAN-DTA/models/model_4.pth'
    ]
    criterion = torch.nn.MSELoss()

    def mean_std(CI, RMSE, MAE, SD, CORR):
        CI_mean, CI_std = np.mean(CI, axis=0), np.std(CI, axis=0)
        RMSE_mean, RMSE_std = np.mean(RMSE, axis=0), np.std(RMSE, axis=0)
        MAE_mean, MAE_std = np.mean(MAE, axis=0), np.std(MAE, axis=0)
        SD_mean, SD_std = np.mean(SD, axis=0), np.std(SD, axis=0)
        CORR_mean, CORR_std = np.mean(CORR, axis=0), np.std(CORR, axis=0)
        print(f'{CI_mean[0]:.3f}' + '\t' + f'{CI_std[0]:.3f}' + '\t'
              + f'{RMSE_mean[0]:.3f}' + '\t' + f'{RMSE_std[0]:.3f}' + '\t'
              + f'{MAE_mean[0]:.3f}' + '\t' + f'{MAE_std[0]:.3f}' + '\t'
              + f'{SD_mean[0]:.3f}' + '\t' + f'{SD_std[0]:.3f}' + '\t'
              + f'{CORR_mean[0]:.3f}' + '\t' + f'{CORR_std[0]:.3f}' + '\t')
    repeats = args.get('repeat')
    CI_re, RMSE_re, MAE_re, SD_re, CORR_re = [], [], [], [], []
    CI_2013, RMSE_2013, MAE_2013, SD_2013, CORR_2013 = [], [], [], [], []
    CI_2016, RMSE_2016, MAE_2016, SD_2016, CORR_2016 = [], [], [], [], []
    CI_hiq, RMSE_hiq, MAE_hiq, SD_hiq, CORR_hiq = [], [], [], [], []
    for repeat in range(repeats):
        checkpoint = torch.load(model_list[repeat], map_location=device)
        model.load_state_dict(checkpoint['model'])

        test_evaluation = test(model, dataloader, criterion, device)
        test2013_evaluation = test(model, dataloader2013, criterion, device)
        test2016_evaluation = test(model, dataloader2016, criterion, device)
        test_hiq_evaluation = test(model, dataloader_hiq, criterion, device)

        CI_re.append(test_evaluation['c_index'])
        RMSE_re.append(test_evaluation['RMSE'])
        MAE_re.append(test_evaluation['MAE'])
        SD_re.append(test_evaluation['SD'])
        CORR_re.append(test_evaluation['CORR'])

        CI_2013.append(test2013_evaluation['c_index'])
        RMSE_2013.append(test2013_evaluation['RMSE'])
        MAE_2013.append(test2013_evaluation['MAE'])
        SD_2013.append(test2013_evaluation['SD'])
        CORR_2013.append(test2013_evaluation['CORR'])

        CI_2016.append(test2016_evaluation['c_index'])
        RMSE_2016.append(test2016_evaluation['RMSE'])
        MAE_2016.append(test2016_evaluation['MAE'])
        SD_2016.append(test2016_evaluation['SD'])
        CORR_2016.append(test2016_evaluation['CORR'])

        CI_hiq.append(test_hiq_evaluation['c_index'])
        RMSE_hiq.append(test_hiq_evaluation['RMSE'])
        MAE_hiq.append(test_hiq_evaluation['MAE'])
        SD_hiq.append(test_hiq_evaluation['SD'])
        CORR_hiq.append(test_hiq_evaluation['CORR'])

    CI, RMSE, MAE, SD, CORR = np.vstack(CI_re), np.vstack(RMSE_re), np.vstack(MAE_re), np.vstack(SD_re), np.vstack(CORR_re)
    ci_2013, rmse_2013, mae_2013, sd_2013, corr_2013 = np.vstack(CI_2013), np.vstack(RMSE_2013), np.vstack(
        MAE_2013), np.vstack(SD_2013), np.vstack(CORR_2013)
    ci_2016, rmse_2016, mae_2016, sd_2016, corr_2016 = np.vstack(CI_2016), np.vstack(RMSE_2016), np.vstack(
        MAE_2016), np.vstack(SD_2016), np.vstack(CORR_2016)
    ci_hiq, rmse_hiq, mae_hiq, sd_hiq, corr_hiq = np.vstack(CI_hiq), np.vstack(RMSE_hiq), np.vstack(MAE_hiq), np.vstack(
        SD_hiq), np.vstack(CORR_hiq)

    mean_std(CI, RMSE, MAE, SD, CORR)
    mean_std(ci_2013, rmse_2013, mae_2013, sd_2013, corr_2013)
    mean_std(ci_2016, rmse_2016, mae_2016, sd_2016, corr_2016)
    mean_std(ci_hiq, rmse_hiq, mae_hiq, sd_hiq, corr_hiq)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cfg = 'TestConfig'
    config = Config(cfg)
    main(config)
