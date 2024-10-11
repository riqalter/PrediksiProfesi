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
- [Notebook WGAN komplit dengan struktur dataset baru (NotebookVer2-torch-rework) Pytorch](NotebookVer2-torch-rework.ipynb) *update
- [Notebook Komparasi dengan nonDL dengan struktur dataset baru (Notebook-nonDL-rework)](Notebook-nonDL-rework.ipynb)  *update
- [Notebook Conditional WGAN-GP (CWGAN-GP_rewrite) Pytorch](CWGAN-GP_rewrite.ipynb) *BARU

#

<br>

### Hyperparameter (CWGAN-GP_rewrite) Per 11 Oktober 2024:
Merapihkan kode dari CWGAN-RealFakedanPrediksi.ipynb ke NotebookCWGAN-GP_rewrite.ipynb dan melakukan experiment terhadap generator, discriminator dan hyperparameters.

```python
latent_dim = 100      # Dimensi ruang laten untuk generator
num_classes = 8       # Jumlah kelas bidang profesi
data_dim = 6         # Dimensi data input (jumlah fitur)
lambda_gp = 10      # Koefisien gradient penalty untuk WGAN-GP
lr_g = 0.00001       # Learning rate untuk optimizer generator
lr_d = 0.00005       # Learning rate untuk optimizer discriminator
betas = (0.5, 0.9)    # Parameter beta1 dan beta2 untuk Adam optimizer
batch_size = 64      # Ukuran batch data untuk pelatihan
num_epochs = 100   # Jumlah epoch pelatihan
patience =   20      # Jumlah epoch tanpa peningkatan untuk early stopping
min_delta = 0.001    # Minimum peningkatan loss untuk early stopping
```
Perubahan pada Generator dan Discriminator:

Pada Generator:
1. **Blok Residual**: menambahkan koneksi residual yang membantu melatih jaringan yang lebih dalam dengan memungkinkan aliran gradien lebih mudah melalui jaringan.
2. **Desain Modular**: Arsitektur sekarang lebih modular, dengan lapisan input, residual, dan output yang terpisah. Ini membuatnya lebih mudah untuk menyesuaikan kedalaman dan lebar jaringan.
3. **Kedalaman yang Ditingkatkan**: Jaringan sekarang lebih dalam berkat adanya blok residual, yang membantu dalam mempelajari fitur yang lebih kompleks.
4. **Arsitektur Fleksibel**: Jumlah blok residual dapat disesuaikan dengan mudah menggunakan parameter `num_residual_blocks`.
5. **Fungsi Aktivasi yang Lebih Baik**:  menggunakan LeakyReLU dengan kemiringan negatif 0,2, yang dapat membantu mencegah masalah "dying ReLU" dan meningkatkan stabilitas pelatihan.
6. **Operasi Inplace**: Menggunakan `inplace=True` pada fungsi aktivasi dapat menghemat memori selama pelatihan.

Pada Discriminator:
1. **Blok Residual**:  menambahkan blok residual untuk membantu jaringan belajar fitur yang lebih kompleks dan memperlancar aliran gradien.
2. **LayerNorm**: BatchNorm diganti dengan LayerNorm, yang sering kali lebih efektif untuk *discriminator* dalam banyak kasus GAN.
3. **Arsitektur yang Lebih Dalam**: Penambahan blok residual membuat jaringan lebih dalam, sehingga dapat mempelajari representasi yang lebih kaya.
4. **Aktivasi Sigmoid untuk Output Real/Palsu**: Saya menambahkan aktivasi sigmoid di akhir jaringan untuk memastikan output berada dalam rentang [0, 1], sehingga lebih mudah membedakan antara gambar asli dan palsu.
5. **Dropout**: Dropout tetap digunakan untuk regularisasi, yang membantu mencegah *overfitting*.
6. **Fleksibilitas**: Jumlah blok residual dapat disesuaikan dengan mudah menggunakan parameter `num_residual_blocks`.
7. **Aktivasi LeakyReLU**: Saya menggunakan LeakyReLU dengan kemiringan negatif 0,2 untuk mencegah masalah *dying ReLU*.

perbedaan utama antara normalisasi di generator dan discriminator:
1. **Normalisasi**:
   - **Generator** menggunakan **BatchNorm1d**.
   - **Discriminator** menggunakan **LayerNorm**.
2. **Alasan perbedaannya**:
   - **BatchNorm vs LayerNorm**:
     - **BatchNorm** biasanya lebih efektif untuk *generator* karena membantu menjaga stabilitas selama pelatihan dan mengurangi perubahan distribusi internal (*internal covariate shift*).
     - **LayerNorm** lebih cocok untuk *discriminator* karena tidak bergantung pada ukuran batch dan dapat memberikan hasil yang lebih konsisten, terutama saat bekerja dengan batch yang kecil atau bervariasi ukurannya.


---
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
