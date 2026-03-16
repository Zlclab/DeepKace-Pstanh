# DeepKace-Pstanh
## Encoding protein sequences into numerical information.
The dataset about lysine acetylation modification can be found in DeepKace-Pstanh-main/DeepKace-Pstanh/data_fasta/CS_ac.txt

Extract features with the data_code.py script before performing 10-fold cross-validation.
```python
data_code.py
```
## 10-fold cross-validation
The models (DeepKace-Pstanh, DeepKace-ReLU, DeepKace-Tanh et al.) can be obtained by using different activation functions.
The model is stored in the model_dl.py script.
```python
# 10-fold cross-validation: run the training script
data_train.py
```python
# 10-fold cross-validation: use data_train.py
data_train.py





