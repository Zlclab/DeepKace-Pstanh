import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import *
import copy,math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import scipy.stats


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, X):
        # self.encoding
        # [max_len = 512, d_model = 512]

        seq_len = X.shape[1]
        X = X+self.encoding[:seq_len, :]
        # [batch_size = 128, seq_len = 30]

        return X
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class att_lstm(nn.Module):
    def __init__(self,d_model,dk,dv):
        super(att_lstm, self).__init__()
        self.dk = dk
        self.max_len = 201
        self.pos_emb = PositionalEncoding(d_model, self.max_len)

        self.X_change_Q = nn.Linear(in_features=d_model,out_features=dv)
        self.X_change_K = nn.Linear(in_features=d_model,out_features=dv)
        self.X_change_V = nn.Linear(in_features=d_model,out_features=dv)
        self.gate = nn.Linear(in_features=d_model,out_features=dv)
        #self.QKT = lambda x: torch.matmul(x[0], x[1]) / np.sqrt(dk)
        self.attention_score = nn.Softmax(dim=2)


        #self.attention_output = lambda x: torch.matmul(x[0], x[1])


    def forward(self,X):

        X = self.pos_emb(X)

        Q = self.X_change_Q(X)
        K = self.X_change_K(X)
        V = self.X_change_V(X)

        gate = nn.Sigmoid()(self.gate(X))


        QKT = torch.matmul(Q,K.permute(0,2,1)) / np.sqrt(self.dk)

        attention_score = self.attention_score(QKT)

        #attention_score = attention_score - 1 / torch.exp(attention_score)

        attention_output = torch.matmul(attention_score,V)
        attention_output_F = gate*attention_output

        #attention_output_F = torch.add(X,attention_output)
        #attention_output_F = gate*attention_output_F

        #attention_output_F = nn.Dropout(0.3)(attention_output_F)

        return attention_output_F


class tanhLU(nn.Module):
    def __init__(self, alpha=1.0 ,beta=0.0 ,gama=1.0):
        super(tanhLU, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)
        self.gama = torch.nn.Parameter(torch.tensor(gama), requires_grad=True)

    def forward(self, x):
        return self.alpha*torch.tanh(self.gama*x)+ self.beta*x

class Logish(nn.Module):
    def __init__(self):
        super(Logish, self).__init__()

    def forward(self, x):
        return x*torch.log(1+torch.sigmoid(x))


class Pstanh(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(Pstanh, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=True)


    def forward(self, x):
        return torch.where(x<0,self.alpha * x * torch.sigmoid(x) + self.beta * torch.exp(x) * torch.tanh(x),torch.tanh(x))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x*torch.sigmoid(x)

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))


class Result(nn.Module):
    def __init__(self,num):
        super(Result, self).__init__()
        self.Flatten = nn.Flatten()
        self.BatchNorm1d = nn.BatchNorm1d(num_features=11 * 64 * num)

        self.Linear1 = nn.Linear(in_features=11 * 64 * num, out_features=21 * 64)

        #self.act_fun = nn.ReLU()
        #self.act_fun = Logish()
        #self.act_fun = nn.Tanh()
        #self.act_fun = tanhLU()
        #self.act_fun = Mish()
        #self.act_fun = Swish()
        self.act_fun = Pstanh()



        self.Dropout = nn.Dropout(p=0.2)
        self.Linear2 = nn.Linear(in_features=21 * 64, out_features=64)
        self.Linear3 = nn.Linear(in_features=64, out_features=32)
        self.Linear4 = nn.Linear(in_features=32, out_features=1)
        self.out = nn.Sigmoid()

    def forward(self,concat):
        concat = self.Flatten(concat)
        concat = self.BatchNorm1d(concat)

        Z = self.Linear1(concat)
        Z = self.act_fun(Z)
        Z = self.Dropout(Z)

        Z = self.Linear2(Z)
        Z = self.act_fun(Z)
        Z = self.Dropout(Z)

        Z = self.Linear3(Z)
        Z = self.act_fun(Z)
        Z = self.Dropout(Z)

        Z = self.Linear4(Z)
        Z = self.out(Z)


        return Z


class DeepKace(nn.Module):

    def __init__(self):
        super(DeepKace, self).__init__()


        self.channels_matching = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.channels_matching1 = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching2 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching3 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching4 = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching5 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching6 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching7 = nn.Sequential(
            nn.Conv1d(in_channels=20, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.channels_matching8 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )


        self.feature_extraction = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=13, stride=1, padding=6),
            nn.PReLU(),
            #nn.AdaptiveMaxPool1d(output_size=1)
            nn.MaxPool1d(2, 2, padding=1)
        )

        self.feature_extraction_lstm1 = nn.LSTM(input_size=16, hidden_size=32, batch_first=True, bidirectional=True)

        self.attention_row = att_lstm(64,32,64)
        self.batch = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_features=21 * 64)
        )

        self.feature_integration = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_features=11*64*3),

            nn.Linear(in_features=11*64*3, out_features=11*64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=21*64, out_features=64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

        self.result = Result(num=3)


    def share(self,X):

        X = self.feature_extraction(X)
        X = X.permute(0, 2, 1)
        X, (_, _) = self.feature_extraction_lstm1(X)

        X = self.attention_row(X)

        return X

    def _forward_impl(self, one_hot, blosum, AAindex, SD, PC, EGB, PAM, Zscales):

        '''
        one_hot = one_hot.float().permute(0, 2, 1)
        one_hot = self.channels_matching1(one_hot)
        one_hot = self.share(one_hot)
        #Z1 = self.feature_integration(one_hot)
        '''

        blosum = blosum.float().permute(0, 2, 1)
        blosum = self.channels_matching2(blosum)
        blosum = self.share(blosum)
        #Z2 = self.feature_integration(blosum)

        '''
        AAindex = AAindex.float().permute(0, 2, 1)
        AAindex = self.channels_matching3(AAindex)
        AAindex = self.share(AAindex)
        #Z3 = self.feature_integration(AAindex)
        '''
        '''
        SD = SD.float().permute(0, 2, 1)
        SD = self.channels_matching4(SD)
        SD = self.share(SD)
        #Z4 = self.feature_integration(SD)
        '''

        PC = PC.float().permute(0, 2, 1)
        PC = self.channels_matching5(PC)
        PC = self.share(PC)
        #Z5 = self.feature_integration(PC)


        EGB = EGB.float().permute(0, 2, 1)
        EGB = self.channels_matching6(EGB)
        EGB = self.share(EGB)
        #Z6 = self.feature_integration(EGB)

        '''
        PAM = PAM.float().permute(0, 2, 1)
        PAM = self.channels_matching7(PAM)
        PAM = self.share(PAM)
        #Z7 = self.feature_integration(PAM)
        '''

        '''
        Zscales = Zscales.float().permute(0, 2, 1)
        Zscales = self.channels_matching8(Zscales)
        Zscales = self.share(Zscales)
        #Z8 = self.feature_integration(Zscales)
        '''

        concat = torch.cat((blosum,EGB,PC), dim=-1)
        #print(concat.shape)
        #Z = self.feature_integration(concat)
        Z = self.result(concat)

        return Z

    def forward(self, one_hot, blosum, AAindex, SD, PC, EGB, PAM, Zscales):
        return self._forward_impl(one_hot, blosum, AAindex, SD, PC, EGB, PAM, Zscales)
