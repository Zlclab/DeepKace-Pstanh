#数据为fasta格式，名字为 0:> 1:UniprotID 2:物种 3：位点 4：修饰类型 5：标签
import numpy as np
import pandas as pd


def read_fasta(file):
    names = []
    labels = []
    seqs = []
    l = 10
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if '>' in line:
            names.append(line)
            if 'pos' in line:
                labels.append(1)
            else:
                labels.append(0)
        else:
            seqs.append(line[20-l:20+l+1])
    return names,labels,seqs

class Data_code():
    def __init__(self,in_path,out_path=None):
        self.in_path = r'data_fasta/'+in_path
        self.names,self.labels,self.seqs = read_fasta(self.in_path)
        self.out_path = out_path
    #########################输出序列编码
    def out_file(self,code,csv_name):
        labels = self.labels
        data_withlabel = []
        name = []
        name.append('class')
        for j in range(code.shape[1]):
            name.append(('V' + str(j + 1)))
        # data_withlabel.append(name)
        for i in range(len(labels)):
            a = list(code[i])
            a.insert(0, labels[i])
            data_withlabel.append(a)
        df_train = pd.DataFrame(data_withlabel, columns=name)
        df_train.to_csv(csv_name + ".csv", index=False)

    #########################独热编码
    def one_hot(self):
        keys = 'ARNDCQEGHILKMFPSTWYVO'

        #生成字典
        n = 21
        I = np.identity(n)
        dict_code = {}
        i=0
        for key in keys:
            dict_code[key] = list(I[i])
            i += 1
        one_hot = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat+dict_code[mod]
            one_hot.append(mat)
        one_hot=np.array(one_hot)
        #################################
        self.out_file(one_hot,'one_hot')
        print(' ______ one_hot is done! ___ the shape is ',one_hot.shape,'_______')
        #return one_hot
    #########################BLOSUM62阵
    def blosum(self):
        blosum62 = {
            'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
            'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
            'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
            'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
            'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
            'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
            'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
            'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
            'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
            'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
            'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
            'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
            'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
            'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
            'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        }
        blosum = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + blosum62[mod]
            blosum.append(mat)
        blosum = np.array(blosum)
        #################################
        self.out_file(blosum,'blosum')
        print(' ______ blosum is done! ___ the shape is ',blosum.shape,'______')
        #return blosum

    #######################################################
    def AAindex(self):
        AAindex_path = r'AAindex.txt'
        AA = 'ARNDCQEGHILKMFPSTWYV'
        with open(AAindex_path,'r') as f:
            lines = f.readlines()
        AAindex_dict = {}
        for line in lines:
            line = line.strip().split('\t')
            if line[0] == 'AccNo':
                for i in range(len(line)-1):
                    AAindex_dict[line[i+1]] = []
            else:
                for i in range(len(line)-1):
                    AAindex_dict[AA[i]].append(line[i+1])
        AAindex_dict['O'] = [0,0,0,0,0,0,0,0,0,0,0,0]
        AAindex = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + AAindex_dict[mod]
            AAindex.append(mat)
        AAindex = np.array(AAindex)
        #################################
        self.out_file(AAindex, 'AAindex')
        print(' ______ AAindex is done! ___ the shape is ', AAindex.shape, '______')
        # return AAindex

    ##########################################
    def SD(self):
        SD_dict = {
            'A': [1,0,0,0,0,0,0],  # A
            'R': [0,0,0,0,1,0,0],  # R
            'N': [0,0,0,1,0,0,0],  # N
            'D': [0,0,0,0,0,1,0],  # D
            'C': [0,0,0,0,0,0,1],  # C
            'Q': [0,0,0,1,0,0,0],  # Q
            'E': [0,0,0,0,0,1,0],  # E
            'G': [1,0,0,0,0,0,0],  # G
            'H': [0,0,0,1,0,0,0],  # H
            'I': [0,1,0,0,0,0,0],  # I
            'L': [0,1,0,0,0,0,0],  # L
            'K': [0,0,0,0,1,0,0],  # K
            'M': [0,0,1,0,0,0,0],  # M
            'F': [0,1,0,0,0,0,0],  # F
            'P': [0,1,0,0,0,0,0],  # P
            'S': [0,0,1,0,0,0,0],  # S
            'T': [0,0,1,0,0,0,0],  # T
            'W': [0,0,0,1,0,0,0],  # W
            'Y': [0,0,1,0,0,0,0],  # Y
            'V': [1,0,0,0,0,0,0],  # V
            'O': [0,0,0,0,0,0,0],  # O
        }
        SD = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + SD_dict[mod]
            SD.append(mat)
        SD = np.array(SD)
        #################################
        self.out_file(SD, 'SD')
        print(' ______ SD is done! ___ the shape is ', SD.shape, '______')

    def PC(self):
        PC_dict = {
            'A': [1, 0, 0, 0, 0],  # A
            'R': [0, 0, 1, 0, 0],  # R
            'N': [0, 0, 0, 0, 1],  # N
            'D': [0, 0, 0, 1, 0],  # D
            'C': [0, 0, 0, 0, 1],  # C
            'Q': [0, 0, 0, 0, 1],  # Q
            'E': [0, 0, 0, 1, 0],  # E
            'G': [1, 0, 0, 0, 0],  # G
            'H': [0, 0, 1, 0, 0],  # H
            'I': [1, 0, 0, 0, 0],  # I
            'L': [1, 0, 0, 0, 0],  # L
            'K': [0, 0, 1, 0, 0],  # K
            'M': [1, 0, 0, 0, 0],  # M
            'F': [0, 1, 0, 0, 0],  # F
            'P': [0, 0, 0, 0, 1],  # P
            'S': [0, 0, 0, 0, 1],  # S
            'T': [0, 0, 0, 0, 1],  # T
            'W': [0, 1, 0, 0, 0],  # W
            'Y': [0, 1, 0, 0, 0],  # Y
            'V': [1, 0, 0, 0, 0],  # V
            'O': [0, 0, 0, 0, 0],  # O
        }
        PC = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + PC_dict[mod]
            PC.append(mat)
        PC = np.array(PC)
        #################################
        self.out_file(PC, 'PC')
        print(' ______ PC is done! ___ the shape is ', PC.shape, '______')

    def EGB(self):
        EGB_dict = {
            'A': [1, 1, 1],  # A
            'R': [0, 0, 1],  # R
            'N': [1, 0, 0],  # N
            'D': [0, 1, 0],  # D
            'C': [1, 0, 0],  # C
            'Q': [1, 0, 0],  # Q
            'E': [0, 1, 0],  # E
            'G': [1, 1, 1],  # G
            'H': [0, 0, 1],  # H
            'I': [1, 1, 1],  # I
            'L': [1, 1, 1],  # L
            'K': [0, 0, 1],  # K
            'M': [1, 1, 1],  # M
            'F': [1, 1, 1],  # F
            'P': [1, 1, 1],  # P
            'S': [1, 0, 0],  # S
            'T': [1, 0, 0],  # T
            'W': [1, 1, 1],  # W
            'Y': [1, 0, 0],  # Y
            'V': [1, 1, 1],  # V
            'O': [0, 0, 0],  # O
        }
        EGB = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + EGB_dict[mod]
            EGB.append(mat)
        EGB = np.array(EGB)
        #################################
        self.out_file(EGB, 'EGB')
        print(' ______ EGB is done! ___ the shape is ', EGB.shape, '______')

    def PAM(self):
        PAM250 = {
            'A': [2,-2,0,0,-3,1,-1,-1,-1,-2,-1,0,1,0,-2,1,1,0,-6,-3],  #A
            'R': [-2,-4,-1,-1,-4,-3,2,-2,3,-3,0,0,0,1,6,0,-1,-2,2,-4],  # R
            'N': [0,-4,2,1,-3,0,2,-2,1,-3,-2,2,0,1,0,1,0,-2,-4,-2],  # N
            'D': [0,-5,4,3,-6,1,1,-2,0,-4,-3,2,-1,2,-1,0,0,-2,-7,-4],  # D
            'C': [-2,12,-5,-5,-4,-3,-3,-2,-5,-6,-5,-4,-3,-5,-4,0,-2,-2,-8,0],  # C
            'Q': [0,-5,2,2,-5,-1,3,-2,1,-2,-1,1,0,4,1,-1,-1,-2,-5,-4],  # Q
            'E': [0,-5,3,4,-5,0,1,-2,0,-3,-2,1,-1,2,-1,0,0,-2,-7,-4],  # E
            'G': [1,-3,1,0,-5,5,-2,-3,-2,-4,-3,0,0,-1,-3,1,0,-1,-7,-5],  # G
            'H': [-1,-3,1,1,-2,-2,6,-2,0,-2,-2,2,0,3,2,-1,-1,-2,-3,0],  # H
            'I': [-1,-2,-2,-2,1,-3,-2,5,-2,2,2,-2,-2,-2,-2,-1,0,4,-5,-1],  # I
            'L': [-2,-6,-4,-3,2,-4,-2,2,-3,6,4,-3,-3,-2,-3,-3,-2,2,-2,-1],  # L
            'K': [-1,-5,0,0,-5,-2,0,-2,5,-3,0,1,-1,1,3,0,0,-2,-3,-4],  # K
            'M': [-1,-5,-3,-2,0,-3,-2,2,0,4,6,-2,-2,-1,0,-2,-1,2,-4,-2],  # M
            'F': [-3,-4,-6,-5,9,-5,-2,1,-5,2,0,-3,-5,-5,-4,-3,-3,-1,0,7],  # F
            'P': [1,-3,-1,-1,-5,0,0,-2,-1,-3,-2,0,6,0,0,1,0,-1,-6,-5],  # P
            'S': [1,0,0,0,-3,1,-1,-1,0,-3,-2,1,1,-1,0,2,1,-1,-2,-3],  # S
            'T': [1,-2,0,0,-3,0,-1,0,0,-2,-1,0,0,-1,-1,1,3,0,-5,-3],  # T
            'W': [-6,-8,-7,-7,0,-7,-3,-5,-3,-2,-4,-4,-6,-5,2,-2,-5,-6,17,0],  # W
            'Y': [-3,0,-4,-4,7,-5,0,-1,-4,-1,-2,-2,-5,-4,-4,-3,-3,-2,0,10],  # Y
            'V': [0,-2,-2,-2,-1,-1,-2,4,-2,2,2,-2,-1,-2,-2,-1,0,4,-6,-2],  # V
            'O': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],  # O
        }
        PAM = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + PAM250[mod]
            PAM.append(mat)
        PAM = np.array(PAM)
        #################################
        self.out_file(PAM, 'PAM')
        print(' ______ PAM is done! ___ the shape is ', PAM.shape, '______')

    def Zscales(self):
        zscale = {
            'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
            'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
            'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
            'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
            'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
            'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
            'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
            'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
            'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
            'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
            'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
            'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
            'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
            'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
            'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
            'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
            'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
            'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
            'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
            'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
            'O': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
        }
        Zscales = []
        for seq in self.seqs:
            mat = []
            for mod in seq:
                mat = mat + zscale[mod]
            Zscales.append(mat)
        Zscales = np.array(Zscales)
        #################################
        self.out_file(Zscales, 'Zscales')
        print(' ______ Zscales is done! ___ the shape is ', Zscales.shape, '______')

    def forward(self):
        self.one_hot()
        self.blosum()
        self.AAindex()
        self.SD()
        self.PC()
        self.EGB()
        self.PAM()
        self.Zscales()



if __name__ == '__main__':

    data = Data_code(in_path='CS_ac.txt')
    #names,seqs = data.data()
    #one_hot = data.one_hot()
    #blosum = data.blosum()
    #data.one_hot()
    #data.blosum()
    data.forward()
