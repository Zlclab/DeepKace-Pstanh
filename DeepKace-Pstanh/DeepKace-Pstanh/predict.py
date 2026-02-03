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




class earlystopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_cv.pt', trace_func=print):
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
        torch.save(model.state_dict(), self.path)
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
    def __init__(self, model, stop,  model_name='DeepKace'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1)
        #self.optimizer = optim.ASGD(self.model.parameters(), lr=1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5)
        self.loss_function = nn.BCELoss()
        self.early_stopping = stop
        self.batch_size = 200
        self.epochs = 100
        self.seed = 0


    def learn(self, TrainLoader, ValidateLoader):
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            epoch_train_loss = 0.0
            batch_count = 0

            for data in ProgressBar:
                self.optimizer.zero_grad()
                ProgressBar.set_description(f"Epoch {epoch}")

                one_hot, blosum, label, AAindex, SD, PC, EGB, PAM, Zscales = data

                label = label.unsqueeze(1)


                output = self.model(
                    one_hot=one_hot.to(self.device, dtype=torch.float32),
                    blosum=blosum.to(self.device, dtype=torch.float32),
                    AAindex=AAindex.to(self.device, dtype=torch.float32),
                    SD=SD.to(self.device, dtype=torch.float32),
                    PC=PC.to(self.device, dtype=torch.float32),
                    EGB=EGB.to(self.device, dtype=torch.float32),
                    PAM=PAM.to(self.device, dtype=torch.float32),
                    Zscales=Zscales.to(self.device, dtype=torch.float32)
                )
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                epoch_train_loss += loss.item()
                batch_count += 1
                loss.backward()
                self.optimizer.step()


            valid_loss = []
            self.model.eval()
            with torch.no_grad():
                for valid_data in ValidateLoader:

                    valid_one_hot, valid_blosum, valid_labels, valid_AAindex, valid_SD, valid_PC, valid_EGB, valid_PAM, valid_Zscales = valid_data

                    valid_labels = valid_labels.unsqueeze(1)


                    valid_output = self.model(
                        one_hot=valid_one_hot.to(self.device, dtype=torch.float32),
                        blosum=valid_blosum.to(self.device, dtype=torch.float32),
                        AAindex=valid_AAindex.to(self.device, dtype=torch.float32),
                        SD=valid_SD.to(self.device, dtype=torch.float32),
                        PC=valid_PC.to(self.device, dtype=torch.float32),
                        EGB=valid_EGB.to(self.device, dtype=torch.float32),
                        PAM=valid_PAM.to(self.device, dtype=torch.float32),
                        Zscales=valid_Zscales.to(self.device, dtype=torch.float32)
                    )
                    valid_labels = valid_labels.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)
                print(f"valid_loss_avg:{valid_loss_avg:.6f}")


                self.early_stopping(valid_loss_avg, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break

    def inference(self, TestLoader):
        self.model.load_state_dict(torch.load(self.early_stopping.path))

        predicted_value = []
        ground_label = []
        self.model.eval()

        with torch.no_grad():
            for data in TestLoader:
                one_hot, blosum, label, AAindex, SD, PC, EGB, PAM, Zscales = data

                label = label.unsqueeze(1)

                output = self.model(
                    one_hot=one_hot.to(self.device, dtype=torch.float32),
                    blosum=blosum.to(self.device, dtype=torch.float32),
                    AAindex=AAindex.to(self.device, dtype=torch.float32),
                    SD=SD.to(self.device, dtype=torch.float32),
                    PC=PC.to(self.device, dtype=torch.float32),
                    EGB=EGB.to(self.device, dtype=torch.float32),
                    PAM=PAM.to(self.device, dtype=torch.float32),
                    Zscales=Zscales.to(self.device, dtype=torch.float32)
                )
                predicted_value.extend(output.detach().cpu().numpy().flatten().tolist())
                ground_label.extend(label.detach().cpu().numpy().flatten().tolist())

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        pred = np.array(predicted_value)
        pred_round = pred.round()
        true = np.array(ground_label)

        sn = metrics.recall_score(y_pred=pred_round, y_true=true, pos_label=1)
        sp = metrics.recall_score(y_pred=pred_round, y_true=true, pos_label=0)
        mcc = metrics.matthews_corrcoef(y_pred=pred_round, y_true=true)
        acc = metrics.accuracy_score(y_pred=pred_round, y_true=true)
        auroc = metrics.roc_auc_score(y_score=pred, y_true=true)
        f1 = metrics.f1_score(y_pred=pred_round, y_true=true)
        ap = metrics.average_precision_score(y_true=true, y_score=pred)
        log_loss = metrics.log_loss(y_true=true, y_pred=pred)
        kappa = metrics.cohen_kappa_score(y1=pred_round, y2=true)

        return sn, sp, mcc, acc, auroc, f1, ap, log_loss, kappa




def main():
    L = 21

    data1_train = pd.read_csv('one_hot_train.csv')
    X1_train = np.array(data1_train.iloc[:, 1:]).reshape((-1, L, 21)).astype(np.float32)
    Y_train = np.array(data1_train.iloc[:, 0]).reshape((-1, 1)).astype(np.float32)

    data2_train = pd.read_csv('blosum_train.csv')
    X2_train = np.array(data2_train.iloc[:, 1:]).reshape((-1, L, 20)).astype(np.float32)

    data3_train = pd.read_csv('AAindex_train.csv')
    X3_train = np.array(data3_train.iloc[:, 1:]).reshape((-1, L, 12)).astype(np.float32)

    data4_train = pd.read_csv('SD_train.csv')
    X4_train = np.array(data4_train.iloc[:, 1:]).reshape((-1, L, 7)).astype(np.float32)

    data5_train = pd.read_csv('PC_train.csv')
    X5_train = np.array(data5_train.iloc[:, 1:]).reshape((-1, L, 5)).astype(np.float32)

    data6_train = pd.read_csv('EGB_train.csv')
    X6_train = np.array(data6_train.iloc[:, 1:]).reshape((-1, L, 3)).astype(np.float32)

    data7_train = pd.read_csv('PAM_train.csv')
    X7_train = np.array(data7_train.iloc[:, 1:]).reshape((-1, L, 20)).astype(np.float32)

    data8_train = pd.read_csv('Zscales_train.csv')
    X8_train = np.array(data8_train.iloc[:, 1:]).reshape((-1, L, 5)).astype(np.float32)

    # Test set
    data1_test = pd.read_csv('one_hot_test.csv')
    X1_test = np.array(data1_test.iloc[:, 1:]).reshape((-1, L, 21)).astype(np.float32)
    Y_test = np.array(data1_test.iloc[:, 0]).reshape((-1, 1)).astype(np.float32)

    data2_test = pd.read_csv('blosum_test.csv')
    X2_test = np.array(data2_test.iloc[:, 1:]).reshape((-1, L, 20)).astype(np.float32)

    data3_test = pd.read_csv('AAindex_test.csv')
    X3_test = np.array(data3_test.iloc[:, 1:]).reshape((-1, L, 12)).astype(np.float32)

    data4_test = pd.read_csv('SD_test.csv')
    X4_test = np.array(data4_test.iloc[:, 1:]).reshape((-1, L, 7)).astype(np.float32)

    data5_test = pd.read_csv('PC_test.csv')
    X5_test = np.array(data5_test.iloc[:, 1:]).reshape((-1, L, 5)).astype(np.float32)

    data6_test = pd.read_csv('EGB_test.csv')
    X6_test = np.array(data6_test.iloc[:, 1:]).reshape((-1, L, 3)).astype(np.float32)

    data7_test = pd.read_csv('PAM_test.csv')
    X7_test = np.array(data7_test.iloc[:, 1:]).reshape((-1, L, 20)).astype(np.float32)

    data8_test = pd.read_csv('Zscales_test.csv')
    X8_test = np.array(data8_test.iloc[:, 1:]).reshape((-1, L, 5)).astype(np.float32)

    # Create datasets
    Train_Set = SSDataset(X1_train, Y_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train)
    Test_Set = SSDataset(X1_test, Y_test, X2_test, X3_test, X4_test, X5_test, X6_test, X7_test, X8_test)

    val_size = int(0.1 * len(Train_Set))

    # Randomly split the training set into: training subset (90%) + validation subset (10%)
    train_subset, val_subset = torch.utils.data.random_split(
        Train_Set,
        [len(Train_Set) - val_size, val_size]
    )

    Train_Loader = loader.DataLoader(train_subset, batch_size=200, shuffle=True, num_workers=0, drop_last=False)
    Val_Loader = loader.DataLoader(val_subset, batch_size=200, shuffle=False, num_workers=0, drop_last=False)
    Test_Loader = loader.DataLoader(Test_Set, batch_size=1, shuffle=False, num_workers=0)


    early_stopping = earlystopping(patience=10, verbose=True, path='checkpoint_final.pt')
    trainer = Constructor(model=DeepKace(), stop=early_stopping)
    trainer.learn(Train_Loader, Val_Loader)

    pred, true = trainer.inference(Test_Loader)
    sn, sp, mcc, acc, auroc, f1, ap, log_loss, kappa = trainer.measure(pred, true)

    print("\n===== Test set results =====")
    print(f"Sn: {sn:.6f}, Sp: {sp:.6f}, MCC: {mcc:.6f}, Acc: {acc:.6f}, AUROC: {auroc:.6f}")
    print(f"F1: {f1:.6f}, AP: {ap:.6f}, Log_Loss: {log_loss:.6f}, Kappa: {kappa:.6f}")

    # Save results
    df = pd.DataFrame({'predict': pred, 'label': true})
    df.to_csv("test_results.csv", index=False)
    torch.save(trainer.model.state_dict(), 'final_model.pt')


if __name__ == '__main__':
    main()