# PROJECT: IF3270 Tugas Besar 2 — CNN & RNN/LSTM

## META
- course: IF3270 Pembelajaran Mesin
- deadline: 2026-05-15 (Jumat), submit via Edunex (oleh NIM terkecil)
- stack: Python 3.11, NumPy, TensorFlow/Keras, Pillow, NLTK
- target: Semua spesifikasi utama + semua bonus
- paradigm: Modular OOP — setiap layer adalah objek independen dengan `forward()`, `backward()`, `set_weights()`
- team: 3 orang, tidak lintas kelas
- repo structure: `src/` (code + notebooks), `doc/` (PDF report), `README.md`

---

## OBJECTIVES

Implementasi forward propagation CNN, SimpleRNN, dan LSTM **from scratch** (NumPy only).
Eksplorasi pipeline **image captioning** via arsitektur encoder-decoder (CNN + LSTM/RNN).
Validasi implementasi scratch dengan membandingkan output-nya terhadap Keras.

---

## TASK A — CNN (Intel Image Classification)

### Dataset
- Intel Image Classification (~25.000 gambar, 6 kelas: buildings, forest, glacier, mountain, sea, street)
- Split: train / validation / test sudah tersedia di Kaggle

### Architecture Requirements
Model CNN minimal harus memiliki:
- `Conv2D` — shared parameters (standard cross-correlation)
- `LocallyConnected2D` — non-shared parameters (tiap region punya filter sendiri)
- Pooling layers (Max atau Average)
- Flatten / Global Pooling
- Dense layer
- Loss: Sparse Categorical Crossentropy | Optimizer: Adam | Metric: **macro F1-score**

### Data flow
`Tensor(B,C,H,W) → Conv2D → Pooling → Flatten → Dense → Softmax`

### Part 1 — Utility Functions (`utils/image_utils.py`)
Implementasi menggunakan PIL/Pillow + NumPy (tanpa Keras preprocessing):
- `load_image(path, target_size)`: PIL.Image.open → resize → numpy array → normalize [0,1]
- `load_batch(paths, target_size)`: batch loader → output shape `(N, H, W, C)`
- `extract_features(paths, encoder)`: Keras CNN encoder (frozen) → feature vectors → save `.npy`

### Part 2 — From-Scratch Layers (`nn/layers/`)
Semua layer HARUS support `set_weights(keras_weights)` dan method `forward(x: np.ndarray) → np.ndarray`.

| Layer | Bobot Keras | Operasi |
|---|---|---|
| `Conv2D` | kernel `[kH,kW,C_in,C_out]`, bias | Cross-correlation: tiap posisi (i,j), filter k → dot product patch × kernel[k] + bias[k] → aktivasi |
| `LocallyConnected2D` | kernel `[out_rows*out_cols, kH*kW*C_in, C_out]`, bias | Sama seperti Conv2D tapi tanpa parameter sharing |
| `MaxPooling2D` / `AvgPooling2D` | — | Sliding window (pool_size, strides) → max atau avg per channel |
| `GlobalAvgPooling2D` / `GlobalMaxPooling2D` | — | Reduksi seluruh feature map → 1 nilai per channel |
| `Flatten` | — | Reshape `(H,W,C)` → 1D, row-major (C order), konsisten dengan Keras |
| `Dense` | (reuse Tubes 1, tambah `set_weights()`) | `output = activation(W·x + b)` |
| Activations | — | ReLU, Softmax, dan aktivasi lain yang digunakan |

### Part 3 — Keras Training
Latih **dua model** di Keras:
1. **Conv2D model** (shared params) — dijadikan baseline
2. **LocallyConnected2D model** (non-shared params) — posisi layer yang sama

Variasi hyperparameter (hanya untuk Conv2D/shared):

| Variasi | Jumlah |
|---|---|
| Jumlah layer konvolusi | 3 variasi |
| Banyak filter per layer | 3 variasi kombinasi |
| Ukuran filter per layer | 3 variasi kombinasi |
| Jenis pooling (max vs avg) | 2 variasi |

Simpan semua bobot model hasil pelatihan.

### Part 4 — Evaluation
- Load bobot arsitektur terbaik Part 3 → scratch CNN → forward pass → bandingkan macro F1-score vs Keras
- Analisis tiap variasi hyperparameter: prediksi akhir + grafik train/val loss + kesimpulan pengaruh
- Analisis shared vs non-shared: macro F1, jumlah parameter, train/val loss, kesimpulan

---

## TASK B — RNN/LSTM (Image Captioning — Flickr8k)

### Dataset
- Flickr8k: 8.092 gambar, 5 caption/gambar
- Split: 6.000 train / 1.000 val / 1.000 test (Kaggle)

### Architecture: Show and Tell (Vinyals et al., 2015) — Pre-Inject Method
```
Encoder: Pretrained CNN (InceptionV3 atau VGG16, frozen, no top) → 1D feature vector → .npy
Decoder:
  image_feature → Dense projection → embed_dim  (= x_{-1}, timestep t=-1)
  <start> token → Embedding → embed_dim          (= x_0,  timestep t=0)
  RNN/LSTM: h_0 = zeros, c_0 = zeros (LSTM only)
  Dense output → vocab_size → Softmax
Training: teacher forcing
  input  = [CNN_feature, emb(<start>), emb(S_0), ..., emb(S_{N-1})]
  target = [S_0, S_1, ..., S_N]  (shifted by 1)
```

### Keras model input shape
- Caption sequence (prepended with CNN feature): `(seq_len+1, embed_dim)`

### Part 0 — From-Scratch Layers (`nn/layers/`)

| Layer | Operasi |
|---|---|
| `Embedding` | Word ID → lookup table → dense vector |
| `SimpleRNN cell` | `h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)` |
| `LSTM cell` | Forget / input / output gates + cell state; load Keras weights: `kernel`, `recurrent_kernel`, `bias` |
| `Dense projection` | `output = W·x + b` → project ke `embed_dim` untuk x_{-1} |
| `Dense output` | `output = softmax(W·x + b)` → distribusi probabilitas atas vocab |

### Part 1 — Feature Extraction
- Pretrained CNN (InceptionV3 / VGG16), frozen, no classification head, weights=ImageNet
- Forward pass semua gambar Flickr8k → simpan `.npy` (sekali saja, reused oleh RNN dan LSTM)
- Gunakan `load_image()` dari Part 1 CNN

### Part 2 — Caption Preprocessing (`utils/text_utils.py`)
Tools yang diperbolehkan (tidak perlu from scratch):
- `tf.keras.layers.TextVectorization` atau `str.split()` — tokenisasi (kata → integer)
- `str.lower()`, `re.sub()` — lowercase + hapus tanda baca
- `numpy.pad` / `tf.keras.utils.pad_sequences` — padding ke panjang seragam
- `numpy.save` / `json.dump` — simpan vocab dict ke disk
- Special tokens: `<start>`, `<end>`, `<pad>`

### Part 3 — Keras Training (dilakukan 2x: untuk RNN dan untuk LSTM)
Loss: Sparse Categorical Crossentropy | Optimizer: Adam

Variasi (sama untuk RNN dan LSTM):

| Variasi | Jumlah |
|---|---|
| Jumlah recurrent layers | 3 variasi (misal: 1, 2, 3) |
| Ukuran hidden state | 2 variasi (misal: 128, 512) |
| **Total per decoder** | **≥6 variasi** |
| **Total keseluruhan** | **≥12 variasi** |

Simpan semua bobot model hasil pelatihan.

### Part 4 — From-Scratch Architecture
- Load bobot CNN encoder Keras (Part 1) + bobot decoder (Part 3), terpisah untuk RNN dan LSTM
- Rakit 2 arsitektur from scratch: 1 dengan RNN, 1 dengan LSTM
- Pipeline: raw image → CNN encoder (Keras, frozen) → scratch decoder → generated caption

### Part 5 — Experiments & Evaluation

**Step 1:** Jalankan pipeline untuk semua variasi Part 3 → catat BLEU-4 + waktu eksekusi

**Step 2:** Pilih 1 arsitektur terbaik (RNN) dan 1 (LSTM) → jalankan Keras equivalent → catat score + waktu

**Step 3:** Pilih 1 arsitektur terbaik di antara 4 kombinasi (LSTM vs RNN × Keras vs Scratch) → variasi max caption length (≥3 variasi) → catat BLEU-4

**Evaluation axes:**

| Analisis | Yang Dibandingkan | Metrik |
|---|---|---|
| Jumlah layer & hidden state | Semua variasi Part 3 | BLEU-4, METEOR, train/val loss per epoch |
| Keras vs Scratch | Best RNN & best LSTM | Score, waktu eksekusi, train/val loss |
| RNN vs LSTM | Best of each | BLEU-4, waktu, qualitative (≥10 contoh: tinggi/sedang/rendah score + caption + ground truth), analisis vanishing gradient |
| Panjang caption | ≥3 variasi max length | BLEU-4 |

---

## PROJECT TREE

```
src/
├── nn/
│   ├── activations/        # Softmax, ReLU, Sigmoid (reuse Tubes 1)
│   ├── losses/             # CrossEntropy (reuse Tubes 1)
│   ├── initializers/       # Random init (reuse Tubes 1)
│   ├── optimizers/         # Optional (full from-scratch training)
│   ├── layers/
│   │   ├── dense.py        # Modified Tubes 1 + set_weights()
│   │   ├── conv.py         # Conv2D + LocallyConnected2D
│   │   ├── pooling.py      # MaxPool2D, AvgPool2D, GlobalPooling
│   │   ├── flatten.py      # NumPy reshape ke 1D
│   │   ├── embedding.py    # Word ID → vector lookup table
│   │   └── recurrent.py    # SimpleRNN cell + LSTM cell
│   └── models/
│       ├── cnn_model.py    # CNNClassifier builder
│       └── caption_model.py# Encoder-Decoder pre-inject builder
├── utils/
│   ├── image_utils.py      # load_image, load_batch, extract_features (PIL/NumPy)
│   ├── text_utils.py       # caption cleaner, vocab builder, tokenizer, padder
│   └── metrics.py          # BLEU-4, METEOR, caption length stats
└── notebooks/
    ├── 1_cnn_keras_train.ipynb       # Train CNN Keras, semua variasi, save weights
    ├── 2_cnn_scratch_eval.ipynb      # Load weights → scratch CNN forward pass → F1 vs Keras
    ├── 3_caption_keras_train.ipynb   # Train Encoder-Decoder Keras, ≥12 variants, save weights
    └── 4_caption_scratch_eval.ipynb  # Load weights → scratch RNN/LSTM → decode → BLEU-4

doc/
└── report.pdf

README.md
```

---

## REPORT STRUCTURE (`doc/report.pdf`)

```
1. Cover
2. Deskripsi Persoalan
3. Pembahasan
   3.1 Penjelasan Implementasi (kelas, atribut, method)
   3.2 Penjelasan Forward Propagation: CNN | RNN | LSTM
4. Hasil Pengujian
   4.1 CNN
       - Shared vs Non-shared parameter
       - Pengaruh jumlah layer konvolusi
       - Pengaruh banyak filter per layer
       - Pengaruh ukuran filter per layer
       - Pengaruh jenis pooling layer
       - Keras vs From Scratch
   4.2 RNN & LSTM (Image Captioning)
       - Perbandingan jumlah layer: RNN & LSTM
       - Perbandingan ukuran hidden state: RNN & LSTM
       - RNN vs LSTM
       - Keras vs From Scratch
       - Pengaruh panjang maksimum caption terhadap BLEU-4
5. Kesimpulan dan Saran
6. Pembagian Tugas Anggota
7. Referensi
8. Lampiran: Form Penggunaan AI
```

---

## TASK CHECKLIST

### Phase 1 — Setup & Utilities
- [ ] venv Python 3.11 + `requirements.txt`
- [ ] `image_utils.py`: `load_image`, `load_batch` (PIL, resize, normalize [0,1])
- [ ] `image_utils.py`: `extract_features` → InceptionV3/VGG16 → `.npy`
- [ ] `text_utils.py`: lowercase, punctuation cleaner, tokenizer, vocab builder, sequence padder

### Phase 2 — From-Scratch Core (NumPy only)
- [ ] `dense.py`: tambah `set_weights()` (dari Tubes 1)
- [ ] `flatten.py`: NumPy reshape row-major
- [ ] `conv.py`: `Conv2D` (manual cross-correlation)
- [ ] `conv.py`: `LocallyConnected2D` (no shared params)
- [ ] `pooling.py`: `MaxPool2D`, `AvgPool2D`, `GlobalMaxPooling2D`, `GlobalAvgPooling2D`
- [ ] `embedding.py`: word ID → continuous vector
- [ ] `recurrent.py`: `SimpleRNN` cell (tanh)
- [ ] `recurrent.py`: `LSTM` cell (forget/input/output gates + cell state)

### Phase 3 — Model Builders
- [ ] `cnn_model.py`: `CNNClassifier` builder class
- [ ] `caption_model.py`: `ImageCaptioner` encoder-decoder pre-inject builder class

### Phase 4 — Keras Training
- [ ] Notebook 1: CNN training, semua variasi hyperparameter (3+3+3+2), save weights + macro F1
- [ ] Notebook 3: Caption training RNN (≥6 variasi) + LSTM (≥6 variasi), save weights

### Phase 5 — From-Scratch Evaluation
- [ ] Notebook 2: Load Keras weights → scratch CNN → forward pass → macro F1 vs Keras
- [ ] Notebook 4: Load Keras weights → scratch RNN/LSTM → Greedy Decode → BLEU-4 + waktu

### Phase 6 — Bonus (High Priority)
- [ ] **Batch Inference**: refactor semua layer ops untuk support batch dim `(B,...)` via `einsum`/`tensordot`
- [ ] **Backprop**: implement `backward()` untuk Conv2D, MaxPooling, LSTM cell
- [ ] **Init-Inject**: arsitektur alternatif — image feature via Add/Concatenate setelah LSTM; bandingkan BLEU-4 vs pre-inject (Keras + scratch)
- [ ] **Beam Search**: ganti Greedy Decode di Notebook 4 dengan Beam Search (K=3 atau K=5); bandingkan kualitas caption
- [ ] **Grad-CAM**: gradients dari last conv layer → heatmap overlay pada gambar asli; + visualisasi intermediate feature maps

### Phase 7 — Deliverables
- [ ] Tulis seluruh section laporan sesuai struktur di atas
- [ ] Tabel perbandingan CNN semua variasi (Keras vs Scratch, shared vs non-shared, hyperparameter)
- [ ] Tabel evaluasi RNN/LSTM (hyperparameter × BLEU-4 × Keras/Scratch), qualitative analysis ≥10 contoh
- [ ] Kesimpulan + saran
- [ ] Isi Form Log Penggunaan AI (lampiran)
- [ ] `README.md`: deskripsi repo, setup, cara run, pembagian tugas
- [ ] `.gitignore`: exclude dataset & model files (`.h5`, `.npy` besar, dataset dirs)
- [ ] Submit link GitHub via Edunex (oleh NIM terkecil), deadline 15 Mei 2026

---

## KEY CONSTRAINTS

| Constraint | Detail |
|---|---|
| From-scratch library | **NumPy only** untuk forward/backward pass dan data preprocessing CNN |
| No training from scratch (main) | Scratch layers digunakan untuk **inference only**, bukan training |
| Weight loading | Semua scratch layer HARUS support `set_weights(keras_weights)` |
| Encoder frozen | InceptionV3/VGG16 tidak dilatih ulang, hanya feature extractor |
| Pre-inject (default) | Image feature → Dense → embed_dim → diinject sebagai x_{-1} |
| Metric CNN | macro F1-score |
| Metric captioning | BLEU-4 (utama), METEOR (sekunder), waktu eksekusi |
| Min variants CNN | 3 variasi jumlah layer + 3 filter + 3 ukuran filter + 2 pooling type |
| Min variants captioning | ≥6 per decoder (RNN & LSTM), total ≥12 |
| No heavy files in git | Dataset & `.h5` / `.npy` besar tidak boleh di-push ke GitHub |
| Team size | 3 orang, tidak lintas kelas; deadline isi kelompok: 24 April 2026 |
| Academic integrity | Dilarang plagiarisme, kerjasama antar kelompok; pelanggaran → nilai E |

---

## GLOSSARY

| Term | Meaning |
|---|---|
| `nn` | Implementasi layer manual dengan NumPy (bukan Keras) |
| Pre-inject | Image feature diinjeksi ke decoder sebagai x_{-1} (sebelum token `<start>`) |
| Init-inject | Image feature diinjeksi via Add/Concatenate setelah LSTM output (bonus) |
| Shared params | Filter konvolusi yang sama digeser ke seluruh area (Conv2D standar) |
| Non-shared params | Tiap region punya filter sendiri (LocallyConnected2D) |
| Teacher forcing | Input saat training = token ground truth (bukan prediksi sebelumnya) |
| Greedy Decoding | Pilih token dengan probabilitas tertinggi di tiap timestep |
| Beam Search | Eksplorasi K kandidat urutan terbaik secara paralel (bonus, K=3 atau K=5) |
| Grad-CAM | Visualisasi area gambar yang paling diperhatikan model CNN (bonus) |
| BLEU-4 | Metric kualitas caption: n-gram precision overlap hingga n=4 |
| METEOR | Metric kualitas caption: recall-weighted, lebih robust dari BLEU |

---

## KEY REFERENCES

| Resource | Kegunaan |
|---|---|
| Vinyals et al. (2015) — Show and Tell | Arsitektur encoder-decoder utama |
| Tanti et al. (2017) — Where to put the Image | Perbandingan pre-inject vs init-inject |
| Selvaraju et al. (2016) — Grad-CAM | Bonus visualisasi CNN |
| d2l.ai: RNN/LSTM from Scratch | Implementasi recurrent layers |
| d2l.ai: CNN chapter | Implementasi conv layers |
| d2l.ai: Beam Search | Bonus decoder |
| CS231n: Conv Notes | Referensi implementasi Conv2D |
| Keras: Save and Load Models | Load bobot ke scratch |
| Flickr8k dataset (Kaggle) | Dataset captioning |
| Intel Image Classification (Kaggle) | Dataset CNN |