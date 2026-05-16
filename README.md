# Tubes 2 IF3270 Pembelajaran Mesin
**Convolutional Neural Network dan Recurrent Neural Network**

Repository ini memuat hasil pengerjaan Tugas Besar 2 IF3270 Pembelajaran Mesin. Proyek ini difokuskan pada implementasi *Convolutional Neural Network* (CNN), *Simple Recurrent Neural Network* (RNN), dan *Long Short-Term Memory* (LSTM) sepenuhnya dari awal (*from scratch*) dengan memanfaatkan pustaka `NumPy`. Komponen utama *forward propagation* dibangun secara manual untuk menjauhi metode *black box*, lalu diverifikasi tingkat akurasinya menggunakan kerangka kerja Keras.

Dua arsitektur pembelajaran mesin ini dikembangkan dan diuji untuk dua konteks kasus nyata:
1. **Klasifikasi Citra**: Model CNN diimplementasikan untuk mengklasifikasikan kelas gambar memanfaatkan Intel Image Classification Dataset. Termasuk komparasi kinerja algoritma dengan konsep *parameter sharing* (`Conv2D`) vs *non-shared* (`LocallyConnected2D`).
2. **Image Captioning**: Model decoder berbasis RNN/LSTM yang diimplementasikan menggunakan skema *Pre-Inject* untuk menerjemahkan semantik gambar menjadi teks deskriptif menggunakan kombinasi Flickr8k Dataset dan *pretrained* InceptionV3.

## Cara Setup dan Run Program

Aplikasi dan struktur model ditulis menggunakan Python 3.12+ (disesuaikan dengan environment yang Anda miliki). Manajemen pustaka (*dependencies*) dapat dilakukan melalui `pip` maupun `uv`.

### Prasyarat Instalasi
1. Clone repositori ini ke komputer lokal Anda:
   ```bash
   git clone https://github.com/reletz/Tubes2-CNN_RNN-43.git
   cd Tubes2-CNN_RNN-43
   ```
2. Anda bisa menggunakan `uv` untuk instalasi cepat (berdasarkan `pyproject.toml` / `uv.lock`), atau membuat virtual environment standar menggunakan `pip`:
   ```bash
   # Opsi 1: Menggunakan pip standar
   python -m venv .venv
   .venv\Scripts\activate  # Untuk Windows
   pip install -r requirements.txt

   # Opsi 2: Menggunakan uv
   uv sync
   ```

### Menjalankan Notebook Pengujian
Seluruh eksperimen dan hasil evaluasi dikompilasi ke dalam berkas Jupyter Notebook. Anda dapat membukanya menggunakan Jupyter Lab, VSCode, atau IDE lainnya yang mendukung file `.ipynb`.
1. Nyalakan Jupyter Server:
   ```bash
   jupyter notebook
   ```
2. Buka dan jalankan urutan sel di dalam modul eksperimen yang terletak di direktori `src/`:
   - `src/notebook-cnn-10-epoch.ipynb` (Untuk evaluasi model *Convolutional Neural Network*)
   - `src/notebook-rnn-lstm.ipynb` (Untuk evaluasi *Image Captioning* berbasis RNN/LSTM)

*(Pastikan model pre-trained atau dataset eksternal diletakkan secara terstruktur jika diperlukan oleh notebook)*

## Pembagian Tugas Anggota Kelompok

Kelompok 43:

| Nama | NIM | Tugas |
| --- | --- | --- |
| **Andi Farhan Hidayat** | 13523128 | Implementasi Notebook Pengujian CNN dan RNN-LSTM; Membuat laporan |
| **Naufarrel Zhafif Abhista** | 13523149 | Membuat Model CNN, RNN, dan LSTM Scratch; Serta mengimplementasikan bonus; Membuat laporan |
