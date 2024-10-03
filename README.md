# Prediksi profesi pekerjaan mahasiswa menggunakan WGAN


Notebook yang complete :

- [Notebook Data Preposessing](Notebook-Pre.ipynb)
- [Notebook Komparasi dengan nonDL (Notebook-nonDL)](Notebook-nonDL.ipynb) 
- [Notebook Komparasi dengan nonDL (Notebook-nonDL-newdataset)](Notebook-nonDL-newdataset.ipynb) 
- [Notebook WGAN Percobaan Pertama (NotebookVer2) Tensorflow](NotebookVer2.ipynb)
- [Notebook WGAN komplit dengan finetune Percobaan Kedua (NotebookVer2-torch) Pytorch](NotebookVer2-torch.ipynb) 
- [Notebook WGAN komplit dengan finetune Percobaan Ketiga (NotebookVer2-torch-newdataset) Pytorch](NotebookVer2-torch-newdataset.ipynb) 

Notebook yang belum ditest *karena ada masalah dengan software :
- [Notebook WGAN Tensor Komplit dengan finetune (NotebookVer2-1) Tensorflow](NotebookVer2-1.ipynb)

## Notebook Utama:
- [Notebook WGAN komplit dengan finetune Percobaan Ketiga (NotebookVer2-torch-newdataset) Pytorch](NotebookVer2-torch-newdataset.ipynb) 
- [Notebook WGAN komplit dengan struktur dataset baru (NotebookVer2-torch-rework) Pytorch](NotebookVer2-torch-rework.ipynb) *BARU
- [Notebook Komparasi dengan nonDL dengan struktur dataset baru (Notebook-nonDL-rework)](Notebook-nonDL-rework.ipynb)  *BARU

### Perubahan <- (NotebookVer2-torch): (16/09/2024)

- ReLu -> LeakyReLu
- RmsProp -> AdamW
- learning rate (0.0001) -> learning rate (0.00001)

### Perubahan <- (NotebookVer2-torch-newdataset): (27/09/2024) 

(WGAN-GP MODEL)
- input_size 100 -> input_size 128
- hidden_size 128 -> hidden_size 256
- optim.adam -> Adamw + weight_decay=1e-4
- input_size 91 -> input_size 90 (karena perubahan dataset)

### Perubahan <- (NotebookVer2-torch-newdataset): (29/09/2024):
perubahan ini diharapkan untuk meningkatkan stabilitas model

- input_size 90 -> input_size 88 (karena perubahan dataset)
- Implementasi seeding random manual pada setiap model
```python
seed = 42  # Manual seed for reproducibility

# Set manual random seed for reproducibility
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
```
- Implementasi train split (80% train, 20% validation) pada setiap model
- Menambahkan diagram untuk monitoring model seperti MSE dan wassserstein value

<br>

### Hyperparameter (NotebookVer2-torch-newdataset) Per 29 September 2024:
#### 1.Hyperparameter WGAN
```python
# Define hyperparameters
input_size = 128  # Size of the latent vector (noise)
hidden_size = 256
output_size = 88  # Number of features from the dataset (excluding the target variable)
batch_size = 128
epochs = 2500
critic_iterations = 5
weight_clipping_limit = 0.01
lr = 0.00001
seed = 42  # Manual seed
```

### 2.Hyperparameter WGAN-GP
#
```python
# Hyperparameters
input_size = 128  # Size of the noise vector
hidden_size = 256
output_size = 88  # Number of features in the dataset (exclude the target column)
lr = 0.00001  # Learning rate
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

### Hyperparameter (NotebookVer2-torch-rework) Per 03 Oktober 2024:
minor update karena menggunakan struktur dataset baru

#### 1.Hyperparameter WGAN
```python
# Define hyperparameters
input_size = 128  # Size of the latent vector (noise)
hidden_size = 256
output_size = 5  # Number of features from the dataset (excluding the target variable)
batch_size = 128
epochs = 2500
critic_iterations = 5
weight_clipping_limit = 0.01
lr = 0.00001
seed = 42  # Manual seed
```

### 2.Hyperparameter WGAN-GP
#
```python
# Hyperparameters
input_size = 128  # Size of the noise vector
hidden_size = 256
output_size = 5  # Number of features in the dataset (exclude the target column)
lr = 0.00001  # Learning rate
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