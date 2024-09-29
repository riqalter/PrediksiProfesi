# Prediksi profesi pekerjaan mahasiswa menggunakan WGAN


Notebook yang complete :

- [Notebook Data Preposessing](Notebook-Pre.ipynb)
- [Notebook Mencari Model Terbaik dan di Komparasi dengan Pycaret (Notebook-nonDL)](Notebook-nonDL.ipynb) 
- [Notebook Mencari Model Terbaik dan di Komparasi dengan Pycaret (Notebook-nonDL-newdataset)](Notebook-nonDL-newdataset.ipynb) *baru
- [Notebook WGAN Percobaan Pertama (NotebookVer2) Tensorflow](NotebookVer2.ipynb)
- [Notebook WGAN komplit dengan finetune Percobaan Kedua (NotebookVer2-torch) Pytorch](NotebookVer2-torch.ipynb) *diupdate
- [Notebook WGAN komplit dengan finetune Percobaan Ketiga (NotebookVer2-torch-newdataset) Pytorch](NotebookVer2-torch-newdataset.ipynb) *diupdate *baru

Notebook yang belum ditest *karena ada masalah dengan software :
- [Notebook WGAN Tensor Komplit dengan finetune (NotebookVer2-1) Tensorflow](NotebookVer2-1.ipynb)


Perubahan <- (NotebookVer2-torch): (16/09/2024)

- ReLu -> LeakyReLu
- RmsProp -> AdamW
- learning rate (0.0001) -> learning rate (0.00001)

Perubahan <- (NotebookVer2-torch-newdataset): (27/09/2024) 

(WGAN-GP MODEL)
- input_size 100 -> input_size 128
- hidden_size 128 -> hidden_size 256
- optim.adam -> Adamw + weight_decay=1e-4
- input_size 91 -> input_size 90 (karena perubahan dataset)

Hyperparameter (NotebookVer2-torch-newdataset) Per 29 September 2024:
### 1.Hyperparameter WGAN
```python
# Define hyperparameters
input_size =128  # Size of the latent vector (noise)
hidden_size = 256
output_size = 90  # Number of features from the dataset (excluding the target variable)
batch_size = 128
epochs = 2500
critic_iterations = 5
weight_clipping_limit = 0.01
lr = 0.00001
```



### 2.Hyperparameter WGAN-GP

```python
# Hyperparameters
input_size = 128  # Size of the noise vector
hidden_size = 256
output_size = 90  # Number of features in the dataset (exclude the target column)
lr = 0.00001  # 1e-5 Learning rate
batch_size = 128
n_epochs = 2500
n_critic = 5  # Critic steps per generator step
lambda_gp = 10  # Weight for gradient penalty
```
> menggunakan early stopping untuk model WGAN-GP

```python
# Hyperparameters for early stopping
patience = 1000  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change in the monitored loss to qualify as improvement
best_loss = float('inf')  # Initialize best loss with a very large number
early_stop_counter = 0  # Counter to track patience
```