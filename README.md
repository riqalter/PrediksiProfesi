# Prediksi profesi pekerjaan mahasiswa menggunakan WGAN


Notebook yang complete :

- [Notebook Data Preposessing](Notebook-Pre.ipynb)
- [Notebook Mencari Model Terbaik dan di Komparasi dengan Pycaret (Notebook-nonDL)](Notebook-nonDL.ipynb) *baru
- [Notebook WGAN Percobaan Pertama (NotebookVer2) Tensorflow](NotebookVer2.ipynb)
- [Notebook WGAN komplit dengan finetune Percobaan Kedua (NotebookVer2-torch) Pytorch](NotebookVer2-torch.ipynb) *diupdate

Notebook yang belum ditest *karena ada masalah dengan software :
- [Notebook WGAN Tensor Komplit dengan finetune (NotebookVer2-1) Tensorflow](NotebookVer2-1.ipynb)


Perubahan <- (NotebookVer2-torch): (16/09/2024)

- ReLu -> LeakyReLu
- RmsProp -> AdamW
- learning rate (0.0001) -> learning rate (0.00001)

Perubahan <- (NotebookVer2-1): (27/09/2024) 

(WGAN-GP MODEL)
- input_size 100 -> input_size 128
- hidden_size 128 -> hidden_size 256
- optim.adam -> Adamw + weight_decay=1e-4
- input_size 91 -> input_size 90 (karena perubahan dataset)
