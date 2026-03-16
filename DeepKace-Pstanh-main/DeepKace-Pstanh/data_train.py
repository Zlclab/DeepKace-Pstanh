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
from fun import *
from sklearn import metrics
from sklearn import model_selection
import random

import time

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader

class earlystopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class SSDataset(Dataset):

    def __init__(self, completed_one_hot, completed_labels,completed_blosum,completed_AAindex,completed_SD,completed_PC,completed_EGB,completed_PAM,completed_Zscales):
        self.completed_one_hot = completed_one_hot
        self.completed_labels = completed_labels
        self.completed_blosum = completed_blosum
        self.completed_AAindex = completed_AAindex
        self.completed_SD = completed_SD
        self.completed_PC =completed_PC
        self.completed_EGB =completed_EGB
        self.completed_PAM = completed_PAM
        self.completed_Zscales = completed_Zscales


    def __getitem__(self, item):
        return self.completed_one_hot[item], self.completed_blosum[item], self.completed_labels[item], self.completed_AAindex[item], self.completed_SD[item], self.completed_PC[item], self.completed_EGB[item], self.completed_PAM[item], self.completed_Zscales[item]


    def __len__(self):
        return self.completed_one_hot.shape[0]



class Constructor:
    """
        按照不同模型的接收维数，修改相关的样本维数，如：
        特征融合策略不同，卷积操作不同（1D或2D），是否融合形状特征等
    """

    def __init__(self, model, stop, model_name='DeepKace'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        #self.optimizer = optim.Adadelta(self.model.parameters(),lr=1,weight_decay=0.01)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1)
        # self.optimizer = optim.ASGD(self.model.parameters(),lr=1)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=True)
        self.loss_function = nn.BCELoss()
        self.early_stopping = stop

        self.batch_size = 200
        self.epochs = 100
        self.seed = 0

    def learn(self, TrainLoader, ValidateLoader):

        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                one_hot, blosum, label, AAindex, SD, PC, EGB, PAM, Zscales = data
                output = self.model(one_hot=one_hot.to(self.device), blosum=blosum.to(self.device),AAindex=AAindex.to(self.device),SD=SD.to(self.device),PC=PC.to(self.device),EGB=EGB.to(self.device),PAM=PAM.to(self.device),Zscales=Zscales.to(self.device))
                #print(output.shape)
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()

            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_one_hot, valid_blosum, valid_labels, valid_AAindex, valid_SD, valid_PC, valid_EGB, valid_PAM, valid_Zcales in ValidateLoader:
                    valid_output = self.model(one_hot=valid_one_hot.to(self.device), blosum=valid_blosum.to(self.device),AAindex=valid_AAindex.to(self.device),SD=valid_SD.to(self.device),PC=valid_PC.to(self.device),EGB=valid_EGB.to(self.device),PAM=valid_PAM.to(self.device),Zscales=valid_Zcales.to(self.device))
                    valid_labels = valid_labels.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))

                self.scheduler.step(valid_loss_avg)
                print("验证集loss:{}".format(valid_loss_avg))
                self.early_stopping(valid_loss_avg, self.model)
                if self.early_stopping.early_stop:
                    print("此时早停！")
                    break

            # torch.save(self.model.state_dict(), path + '\\' + self.model_name + '.pth')

    def inference(self, TestLoader):

        self.model.load_state_dict(torch.load('checkpoint.pt'))

        predicted_value = []
        ground_label = []
        self.model.eval()

        for one_hot, blosum,label, AAindex, SD, PC, EGB, PAM, Zscales in TestLoader:
            output = self.model(one_hot.to(self.device),blosum.to(self.device),AAindex=AAindex.to(self.device),SD=SD.to(self.device),PC=PC.to(self.device),EGB=EGB.to(self.device),PAM=PAM.to(self.device),Zscales=Zscales.to(self.device))
            """ To scalar"""
            predicted_value.append(output.squeeze(dim=0).detach().cpu().numpy().tolist())
            ground_label.append(label.squeeze(dim=0).detach().cpu().numpy().tolist())

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):

        sn = metrics.recall_score(y_pred=np.array(predicted_value).round(), y_true=np.array(ground_label), pos_label=1)
        sp = metrics.recall_score(y_pred=np.array(predicted_value).round(), y_true=np.array(ground_label), pos_label=0)
        mcc = metrics.matthews_corrcoef(y_pred=np.array(predicted_value).round(), y_true=np.array(ground_label))
        acc = metrics.accuracy_score(y_pred=np.array(predicted_value).round(), y_true=np.array(ground_label))
        auroc = metrics.roc_auc_score(y_score=np.array(predicted_value), y_true=np.array(ground_label))

        print(sn, sp, mcc, acc, auroc)
        return sn, sp, mcc, acc, auroc

    def run(self, Train_Set, Vaild_Set, Test__Set):

        Train_Loader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                         batch_size=self.batch_size, shuffle=True, num_workers=0)
        Vaild_Loader = loader.DataLoader(dataset=Vaild_Set, drop_last=True,
                                         batch_size=self.batch_size, shuffle=False, num_workers=0)

        Test_Loader = loader.DataLoader(dataset=Test__Set,
                                        batch_size=1, shuffle=False, num_workers=0)

        self.learn(Train_Loader, Vaild_Loader)
        predicted_value, ground_label = self.inference(Test_Loader)


        sn, sp, mcc, acc, auroc = self.measure(predicted_value, ground_label)

        return sn, sp, mcc, acc, auroc,predicted_value,ground_label


def main():
    K = 10
    ratio_k = 0.1
    L = 21


    data1 = pd.read_csv('one_hot.csv')
    X1 = np.array(data1.iloc[:, 1:])
    seqs_num = X1.shape[0]
    X1 = X1.reshape((X1.shape[0],L,21))
    Y = np.array(data1.iloc[:, 0])
    Y = Y.reshape((Y.shape[0], 1))

    data2 = pd.read_csv('blosum.csv')
    X2 = np.array(data2.iloc[:, 1:])
    X2 = X2.reshape((X2.shape[0],L,20))

    data3 = pd.read_csv('AAindex.csv')
    X3 = np.array(data3.iloc[:, 1:])
    X3 = X3.reshape((X3.shape[0], L, 12))

    data4 = pd.read_csv('SD.csv')
    X4 = np.array(data4.iloc[:, 1:])
    X4 = X4.reshape((X4.shape[0], L, 7))

    data5 = pd.read_csv('PC.csv')
    X5 = np.array(data5.iloc[:, 1:])
    X5 = X5.reshape((X5.shape[0], L, 5))

    data6 = pd.read_csv('EGB.csv')
    X6 = np.array(data6.iloc[:, 1:])
    X6 = X6.reshape((X6.shape[0], L, 3))

    data7 = pd.read_csv('PAM.csv')
    X7 = np.array(data7.iloc[:, 1:])
    X7 = X7.reshape((X7.shape[0], L, 20))

    data8 = pd.read_csv('Zscales.csv')
    X8 = np.array(data8.iloc[:, 1:])
    X8 = X8.reshape((X8.shape[0], L, 5))

    # k-folds cross-validation
    indices = np.arange(seqs_num)
    np.random.seed(10)
    np.random.shuffle(indices)

    seqs1_data_train = X1[indices]
    seqs2_data_train = X2[indices]
    seqs3_data_train = X3[indices]
    seqs4_data_train = X4[indices]
    seqs5_data_train = X5[indices]
    seqs6_data_train = X6[indices]
    seqs7_data_train = X7[indices]
    seqs8_data_train = X8[indices]


    intensity_train = Y[indices]

    train_ids, test_ids, valid_ids = Id_k_folds(seqs_num, k_folds=K, ratio=ratio_k)
    Sn = []
    Sp = []
    Acc = []
    Mcc = []
    auROC = []
    pre_value = []
    true_label = []
    start_time = time.time()

    for fold in range(K):
        x1_train = seqs1_data_train[train_ids[fold]]
        x2_train = seqs2_data_train[train_ids[fold]]
        x3_train = seqs3_data_train[train_ids[fold]]
        x4_train = seqs4_data_train[train_ids[fold]]
        x5_train = seqs5_data_train[train_ids[fold]]
        x6_train = seqs6_data_train[train_ids[fold]]
        x7_train = seqs7_data_train[train_ids[fold]]
        x8_train = seqs8_data_train[train_ids[fold]]
        y_train = intensity_train[train_ids[fold]]

        x1_valid = seqs1_data_train[valid_ids[fold]]
        x2_valid = seqs2_data_train[valid_ids[fold]]
        x3_valid = seqs3_data_train[valid_ids[fold]]
        x4_valid = seqs4_data_train[valid_ids[fold]]
        x5_valid = seqs5_data_train[valid_ids[fold]]
        x6_valid = seqs6_data_train[valid_ids[fold]]
        x7_valid = seqs7_data_train[valid_ids[fold]]
        x8_valid = seqs8_data_train[valid_ids[fold]]
        y_valid = intensity_train[valid_ids[fold]]

        x1_test = seqs1_data_train[test_ids[fold]]
        x2_test = seqs2_data_train[test_ids[fold]]
        x3_test = seqs3_data_train[test_ids[fold]]
        x4_test = seqs4_data_train[test_ids[fold]]
        x5_test = seqs5_data_train[test_ids[fold]]
        x6_test = seqs6_data_train[test_ids[fold]]
        x7_test = seqs7_data_train[test_ids[fold]]
        x8_test = seqs8_data_train[test_ids[fold]]
        y_test = intensity_train[test_ids[fold]]

        Train_Set = SSDataset(completed_one_hot=x1_train, completed_labels=y_train, completed_blosum=x2_train,completed_AAindex = x3_train,completed_SD = x4_train,completed_PC = x5_train,completed_EGB = x6_train,completed_PAM = x7_train,completed_Zscales = x8_train)
        Vaild_Set = SSDataset(completed_one_hot=x1_valid, completed_labels=y_valid, completed_blosum=x2_valid,completed_AAindex = x3_valid,completed_SD = x4_valid,completed_PC = x5_valid,completed_EGB = x6_valid,completed_PAM = x7_valid,completed_Zscales = x8_valid)
        Test__Set = SSDataset(completed_one_hot=x1_test, completed_labels=y_test, completed_blosum=x2_test,completed_AAindex = x3_test,completed_SD = x4_test,completed_PC = x5_test,completed_EGB = x6_test,completed_PAM = x7_test,completed_Zscales = x8_test)

        early_stopping = earlystopping(patience=10, verbose=True)
        # 根据输入特征应改变特征匹配块的维度
        Train = Constructor(model=DeepKace(), stop=early_stopping)
        # Train = Constructor(model=d_ssca())
        print("\n_______________fold", fold, "_____________\n")
        # 计算运行时间
        sn, sp, mcc, acc, auroc,predicted_value,ground_label = Train.run(Train_Set, Vaild_Set, Test__Set)
        Sn.append(sn)
        Sp.append(sp)
        Mcc.append(mcc)
        Acc.append(acc)
        auROC.append(auroc)
        pre_value += np.array(predicted_value).flatten().tolist()
        true_label += np.array(ground_label).flatten().tolist()

    end_time = time.time()
    total_time = end_time - start_time
    print("total_time:", total_time/K)

    print("the mean Sn is {}. the mean Sp is {}. the mean Mcc is {}. the mean Acc is {}. the mean auROC is {}".format(
        np.mean(Sn), np.mean(Sp), np.mean(Mcc), np.mean(Acc), np.mean(auROC)))
    #print("pre_value:",len(pre_value),pre_value)
    #print("true_label:",len(true_label),true_label)
    name = ['predict','label']
    df = pd.DataFrame(np.transpose((pre_value,true_label)), columns=name)
    df.to_csv("tanhLU.csv", index=False)


if __name__ == '__main__': main()



























