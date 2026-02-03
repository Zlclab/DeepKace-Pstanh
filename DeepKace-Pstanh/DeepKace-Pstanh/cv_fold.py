import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import numpy as np
from tqdm import tqdm
import pandas as pd
from model_dl import DeepKace
from torch.utils.data import Dataset
from sklearn import metrics
import random
import time
from fun import Id_k_folds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




class earlystopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        self.val_loss_min = val_loss


class SSDataset(Dataset):
    def __init__(self, completed_one_hot, completed_labels, completed_blosum, completed_AAindex, completed_SD,
                 completed_PC, completed_EGB, completed_PAM, completed_Zscales):
        self.completed_one_hot = completed_one_hot
        self.completed_labels = completed_labels.squeeze(axis=1)
        self.completed_blosum = completed_blosum
        self.completed_AAindex = completed_AAindex
        self.completed_SD = completed_SD
        self.completed_PC = completed_PC
        self.completed_EGB = completed_EGB
        self.completed_PAM = completed_PAM
        self.completed_Zscales = completed_Zscales

    def __getitem__(self, item):

        return (
            self.completed_one_hot[item],
            self.completed_blosum[item],
            self.completed_labels[item].astype(np.float32),
            self.completed_AAindex[item],
            self.completed_SD[item],
            self.completed_PC[item],
            self.completed_EGB[item],
            self.completed_PAM[item],
            self.completed_Zscales[item]
        )

    def __len__(self):
        return self.completed_one_hot.shape[0]



class Constructor:
    def __init__(self, model, stop, model_name='DeepKace'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5)
        self.loss_function = nn.BCELoss()
        self.early_stopping = stop
        self.batch_size = 200
        self.epochs = 100
        self.seed = 0

    def learn(self, TrainLoader, ValidateLoader, fold):
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description(f"Fold {fold + 1} Epoch {epoch}")
                one_hot, blosum, label, AAindex, SD, PC, EGB, PAM, Zscales = data


                label = label.unsqueeze(1)

                output = self.model(
                    one_hot=one_hot.to(self.device),
                    blosum=blosum.to(self.device),
                    AAindex=AAindex.to(self.device),
                    SD=SD.to(self.device),
                    PC=PC.to(self.device),
                    EGB=EGB.to(self.device),
                    PAM=PAM.to(self.device),
                    Zscales=Zscales.to(self.device)
                )

                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()


            valid_loss = []
            self.model.eval()
            with torch.no_grad():
                for valid_data in ValidateLoader:
                    valid_one_hot, valid_blosum, valid_labels, valid_AAindex, valid_SD, valid_PC, valid_EGB, valid_PAM, valid_Zscales = valid_data

                    valid_labels = valid_labels.unsqueeze(1)

                    valid_output = self.model(
                        one_hot=valid_one_hot.to(self.device),
                        blosum=valid_blosum.to(self.device),
                        AAindex=valid_AAindex.to(self.device),
                        SD=valid_SD.to(self.device),
                        PC=valid_PC.to(self.device),
                        EGB=valid_EGB.to(self.device),
                        PAM=valid_PAM.to(self.device),
                        Zscales=valid_Zscales.to(self.device)
                    )

                    valid_labels = valid_labels.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))

                self.scheduler.step(valid_loss_avg)
                print(f"Fold {fold + 1} c:{valid_loss_avg:.6f}")

                self.early_stopping(valid_loss_avg, self.model)

                if self.early_stopping.early_stop:
                    print(f"Fold {fold + 1} Early stopping triggered!")
                    break

    def inference(self, TestLoader):

        predicted_value = []
        ground_label = []
        self.model.eval()

        with torch.no_grad():
            for data in TestLoader:
                one_hot, blosum, label, AAindex, SD, PC, EGB, PAM, Zscales = data


                label = label.unsqueeze(1)

                output = self.model(
                    one_hot.to(self.device),
                    blosum.to(self.device),
                    AAindex=AAindex.to(self.device),
                    SD=SD.to(self.device),
                    PC=PC.to(self.device),
                    EGB=EGB.to(self.device),
                    PAM=PAM.to(self.device),
                    Zscales=Zscales.to(self.device)
                )

                predicted_value.extend(output.detach().cpu().numpy().flatten().tolist())
                ground_label.extend(label.detach().cpu().numpy().flatten().tolist())

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        pred = np.array(predicted_value)
        pred_round = pred.round()
        true = np.array(ground_label)

        # 计算指标
        sn = metrics.recall_score(y_pred=pred_round, y_true=true, pos_label=1)
        sp = metrics.recall_score(y_pred=pred_round, y_true=true, pos_label=0)
        mcc = metrics.matthews_corrcoef(y_pred=pred_round, y_true=true)
        acc = metrics.accuracy_score(y_pred=pred_round, y_true=true)
        auroc = metrics.roc_auc_score(y_score=pred, y_true=true)

        f1 = metrics.f1_score(y_pred=pred_round, y_true=true)
        ap = metrics.average_precision_score(y_true=true, y_score=pred)
        log_loss = metrics.log_loss(y_true=true, y_pred=pred)
        kappa = metrics.cohen_kappa_score(y1=pred_round, y2=true)

        print(
            f"sn:{sn:.6f}, sp:{sp:.6f}, mcc:{mcc:.6f}, acc:{acc:.6f}, auroc:{auroc:.6f}, f1:{f1:.6f}, ap:{ap:.6f}, log_loss:{log_loss:.6f}, kappa:{kappa:.6f}")
        return sn, sp, mcc, acc, auroc, f1, ap, log_loss, kappa

    def run(self, Train_Set, Valid_Set, fold):

        Train_Loader = loader.DataLoader(dataset=Train_Set, drop_last=True, batch_size=self.batch_size, shuffle=True,
                                         num_workers=0)
        Valid_Loader = loader.DataLoader(dataset=Valid_Set, drop_last=True, batch_size=self.batch_size, shuffle=False,
                                         num_workers=0)
        Test_Loader = loader.DataLoader(dataset=Valid_Set, batch_size=1, shuffle=False, num_workers=0)


        self.learn(Train_Loader, Valid_Loader, fold)
        predicted_value, ground_label = self.inference(Test_Loader)

        sn, sp, mcc, acc, auroc, f1, ap, log_loss, kappa = self.measure(predicted_value, ground_label)

        return sn, sp, mcc, acc, auroc, f1, ap, log_loss, kappa, predicted_value, ground_label


def main():
    K = 10
    L = 21


    data1 = pd.read_csv('one_hot_train.csv')
    X1 = np.array(data1.iloc[:, 1:]).reshape((-1, L, 21))
    Y = np.array(data1.iloc[:, 0]).reshape((-1, 1))

    data2 = pd.read_csv('blosum_train.csv')
    X2 = np.array(data2.iloc[:, 1:]).reshape((-1, L, 20))

    data3 = pd.read_csv('AAindex_train.csv')
    X3 = np.array(data3.iloc[:, 1:]).reshape((-1, L, 12))

    data4 = pd.read_csv('SD_train.csv')
    X4 = np.array(data4.iloc[:, 1:]).reshape((-1, L, 7))

    data5 = pd.read_csv('PC_train.csv')
    X5 = np.array(data5.iloc[:, 1:]).reshape((-1, L, 5))

    data6 = pd.read_csv('EGB_train.csv')
    X6 = np.array(data6.iloc[:, 1:]).reshape((-1, L, 3))

    data7 = pd.read_csv('PAM_train.csv')
    X7 = np.array(data7.iloc[:, 1:]).reshape((-1, L, 20))

    data8 = pd.read_csv('Zscales_train.csv')
    X8 = np.array(data8.iloc[:, 1:]).reshape((-1, L, 5))


    seqs_num = X1.shape[0]
    indices = np.arange(seqs_num)
    np.random.seed(10)
    np.random.shuffle(indices)

    seqs1_data = X1[indices]
    seqs2_data = X2[indices]
    seqs3_data = X3[indices]
    seqs4_data = X4[indices]
    seqs5_data = X5[indices]
    seqs6_data = X6[indices]
    seqs7_data = X7[indices]
    seqs8_data = X8[indices]
    intensity = Y[indices]


    train_ids, valid_ids = Id_k_folds(seqs_num, k_folds=K)


    Sn, Sp, Mcc, Acc, auROC = [], [], [], [], []
    F1, AP, Log_Loss, Kappa = [], [], [], []
    all_pred_list, all_true_list = [], []
    start_time = time.time()


    for fold in range(K):
        print(f"\n=============== Fold {fold + 1}/{K} ===============")


        x1_train = seqs1_data[train_ids[fold]]
        x2_train = seqs2_data[train_ids[fold]]
        x3_train = seqs3_data[train_ids[fold]]
        x4_train = seqs4_data[train_ids[fold]]
        x5_train = seqs5_data[train_ids[fold]]
        x6_train = seqs6_data[train_ids[fold]]
        x7_train = seqs7_data[train_ids[fold]]
        x8_train = seqs8_data[train_ids[fold]]
        y_train = intensity[train_ids[fold]]

        x1_valid = seqs1_data[valid_ids[fold]]
        x2_valid = seqs2_data[valid_ids[fold]]
        x3_valid = seqs3_data[valid_ids[fold]]
        x4_valid = seqs4_data[valid_ids[fold]]
        x5_valid = seqs5_data[valid_ids[fold]]
        x6_valid = seqs6_data[valid_ids[fold]]
        x7_valid = seqs7_data[valid_ids[fold]]
        x8_valid = seqs8_data[valid_ids[fold]]
        y_valid = intensity[valid_ids[fold]]


        Train_Set = SSDataset(x1_train, y_train, x2_train, x3_train, x4_train, x5_train, x6_train, x7_train, x8_train)
        Valid_Set = SSDataset(x1_valid, y_valid, x2_valid, x3_valid, x4_valid, x5_valid, x6_valid, x7_valid, x8_valid)


        early_stopping = earlystopping(patience=10, verbose=True, path=f'checkpoint_fold{fold}.pt')
        trainer = Constructor(model=DeepKace(), stop=early_stopping)
        sn, sp, mcc, acc, auroc, f1, ap, log_loss, kappa, pred, true = trainer.run(Train_Set, Valid_Set, fold)


        Sn.append(sn)
        Sp.append(sp)
        Mcc.append(mcc)
        Acc.append(acc)
        auROC.append(auroc)
        F1.append(f1)
        AP.append(ap)
        Log_Loss.append(log_loss)
        Kappa.append(kappa)

        all_pred_list.append(pred)
        all_true_list.append(true)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"\ntotal time: {total_time:.2f}s, average per fold: {total_time / K:.2f}s")


    metrics_dict = {
        'Sn': Sn, 'Sp': Sp, 'Mcc': Mcc, 'Acc': Acc, 'auROC': auROC,
        'F1': F1, 'AP': AP, 'Log_Loss': Log_Loss, 'Kappa': Kappa
    }

    print("\n===== 10-fold cross-validation results (mean ± std) =====")

    for name, values in metrics_dict.items():
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        print(f"{name}: {mean_val:.6f} ± {std_val:.6f}")


if __name__ == '__main__':
    main()