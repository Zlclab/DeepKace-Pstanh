# DeepKace-Pstanh
## Encoding protein sequences into numerical information.
The dataset about lysine acetylation modification can be found in DeepKace-Pstanh/data_fasta/CS_ac.txt.
```python
python data_code.py
```
## Training and evaluating DeepKace-Pstanh.
The models (DeepKace-Pstanh, DeepKace-ReLU, DeepKace-Tanh et al.) can be obtained by using different activation functions.
```python
python data_train.py
```
## predict_results.csv.
Prediction results of different activation functions.
## dataset.csv.
Training set, Validation set, and Test set are used to compare with other methods on an independent validation set.
