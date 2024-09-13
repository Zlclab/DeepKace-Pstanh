import pandas as pd
import numpy as np

def new_fasta(data):
    with open(data,'r') as f:
        lines=f.readlines()
    new_lines=[]
    for line in lines:
        line=line.strip()
        if line[0]=='>':
            temp=line.split('	')
            if temp[-1]=='pos':
                flag=['1']
            else:
                flag=['0']
            uid=temp[0]+' '+'_'.join(temp[1:3]+flag)+'\n'
            new_lines.append(uid)
        else:
            line=line[10:31]
            new_lines.append(line+'\n')
    with open('CS_ac_new.txt','w') as f:
        f.writelines(new_lines)

def fasta_csv(data):
    with open(data,'r') as f:
        lines=f.readlines()
    unique_id=[]
    seq_21=[]
    label=[]
    for line in lines:
        line=line.strip()

        if line[0]=='>':
            temp=line.split(' ')
            uid = temp[1]
            if uid[-1]=='1':
                flag=1
            else:
                flag=0
            unique_id.append(uid)
            label.append(flag)
        else:
            seq_21.append(line)
    df=pd.DataFrame(
        {
            'unique_id':unique_id,
            'seq_21':seq_21,
            'label':label
        }
    )

    df.to_csv('CS_ac.csv',index=False)


if __name__ == '__main__':
    df='CS_ac.txt'
    new_fasta(df)

    df1='CS_ac_new.txt'
    fasta_csv(df1)
