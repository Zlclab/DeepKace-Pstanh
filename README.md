# DeepKace-Pstanh
## Encoding protein sequences into numerical information.
The dataset about lysine acetylation modification can be found in DeepKace-Pstanh/data_fasta/train_set.txt and DeepKace-Pstanh/data_fasta/test_set.txt.
Extract features with `data_code.py` before training the model.
```python
python data_code.py
```
## Training and testing DeepKace-Pstanh.
The models (DeepKace-Pstanh, DeepKace-ReLU, DeepKace-Tanh et al.) can be obtained by using different activation functions.
```python
# 10-fold cross-validation: use cv_fold.py
python cv_fold.py

# Model testing: use predict.py
python predict.py




