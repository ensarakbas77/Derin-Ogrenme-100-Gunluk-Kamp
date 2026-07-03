# 🍰 PyTorch Structural Setup — Tatlı Sınıflandırma (Dessert Classification)

Bu proje, **PyTorch** kullanılarak sıfırdan bir **Convolutional Neural Network (CNN)** eğiten, modüler ve yeniden kullanılabilir bir bilgisayarlı görü (computer vision) iskeletidir. Amacı; Jupyter Notebook içinde dağınık halde duran deneysel kodu, **her sorumluluğu ayrı bir Python modülüne** taşıyan profesyonel bir proje yapısına dönüştürmektir.

Model, 4 farklı tatlı türünü sınıflandırır:

| Sınıf | Açıklama |
|-------|----------|
| `baklava` | Baklava |
| `cannoli` | Cannoli |
| `cup_cakes` | Cupcake |
| `donuts` | Donut |

> Veri seti, ünlü **Food-101** veri setinden seçilmiş 4 tatlı sınıfının bir alt kümesidir (`desert101`).

---

## 📁 Proje Yapısı

```
PyTorchStructuralSetup/
│
├── main.py                       # 🚀 Giriş noktası: tüm pipeline'ı baştan sona çalıştırır
├── setup_data.py                 # 📦 Veri yükleme: ImageFolder + DataLoader oluşturma
├── model_creation.py             # 🧠 Model mimarisi: DesertClassifier (TinyVGG benzeri CNN)
├── training_testing_engine.py    # 🔁 Eğitim/test döngüleri: train_step, test_step, train
├── utils.py                      # 🛠️ Yardımcılar: model kaydetme + mean/std hesaplama
├── load_model_make_prediction.py # 🔮 Kayıtlı modeli yükleyip tek görsel tahmini yapma
│
├── models/
│   └── desert_classifier.pth     # 💾 Eğitilmiş modelin ağırlıkları (state_dict)
│
├── data/
│   ├── baklava.jpg               # Tek tahmin için örnek görsel
│   └── desert101/
│       ├── train/                # Eğitim verisi (sınıf başına klasör)
│       │   ├── baklava/
│       │   ├── cannoli/
│       │   ├── cup_cakes/
│       │   └── donuts/
│       └── test/                 # Test verisi (aynı klasör yapısı)
│
├── reqierements.txt              # Bağımlılıklar (torch, torchvision)
└── .gitignore
```

---

## 🏗️ Mimari ve Çalışma Mantığı

Proje, klasik **"modüler PyTorch workflow"** yaklaşımını izler. Aşağıdaki diyagram veri ve kontrol akışını gösterir:

```
                 ┌─────────────────────────────────────────────┐
                 │                  main.py                     │
                 │        (tüm süreci orkestra eder)            │
                 └─────────────────────────────────────────────┘
                        │           │            │           │
          ┌─────────────┘           │            │           └─────────────┐
          ▼                         ▼            ▼                         ▼
 ┌─────────────────┐   ┌──────────────────┐  ┌────────────────────┐  ┌──────────────┐
 │  setup_data.py  │   │ model_creation.py│  │ training_testing_  │  │   utils.py   │
 │                 │   │                  │  │     engine.py      │  │              │
 │ DataLoader'lar  │──▶│ DesertClassifier │─▶│ train() döngüsü    │─▶│ save_model() │
 │   + class_names │   │      (CNN)       │  │ (train + test step)│  │  .pth kaydet │
 └─────────────────┘   └──────────────────┘  └────────────────────┘  └──────────────┘
                                                                            │
                                                                            ▼
                                                              models/desert_classifier.pth
                                                                            │
                                                                            ▼
                                                    ┌──────────────────────────────────┐
                                                    │   load_model_make_prediction.py  │
                                                    │  kayıtlı modeli yükle → tahmin    │
                                                    └──────────────────────────────────┘
```

### Her Modülün Görevi

#### 1. `setup_data.py` — Veri Hazırlama Katmanı
`create_dataloaders()` fonksiyonu, klasör-tabanlı görsel veriyi PyTorch'un anlayacağı hale getirir:
- **`datasets.ImageFolder`** ile klasör isimlerini otomatik olarak etiket (label) yapar (ör. `baklava/` → sınıf 0).
- Verilen `transform` pipeline'ını her görsele uygular.
- Eğitim ve test için ayrı **`DataLoader`** nesneleri üretir (eğitimde `shuffle=True`, testte `shuffle=False`).
- `num_workers = os.cpu_count()` ile veri yüklemeyi paralelleştirir.
- Geriye `train_dataloader, test_dataloader, class_names` döndürür.

#### 2. `model_creation.py` — Model Mimarisi
`DesertClassifier`, **TinyVGG** mimarisinden esinlenen bir CNN'dir:

```
Girdi (3 × 64 × 64)
   │
   ▼
┌─ conv_block_1 ─────────────────────────┐
│  Conv2d → ReLU → Conv2d → ReLU         │   → (32 × 64 × 64)
│  MaxPool2d(2)                          │   → (32 × 32 × 32)
└────────────────────────────────────────┘
   │
   ▼
┌─ conv_block_2 ─────────────────────────┐
│  Conv2d → ReLU → Conv2d → ReLU         │   → (32 × 32 × 32)
│  MaxPool2d(2)                          │   → (32 × 16 × 16)
└────────────────────────────────────────┘
   │
   ▼
┌─ classifier ───────────────────────────┐
│  Flatten → Linear                      │   → (4 logit / sınıf sayısı)
└────────────────────────────────────────┘
```

- **`input_shape=3`**: RGB kanal sayısı.
- **`hidden_units=32`**: Her konvolüsyon katmanındaki filtre (feature map) sayısı.
- **`output_shape`**: Sınıf sayısı (burada 4).
- Son katmandaki `hidden_units * 16 * 16` ifadesi, iki `MaxPool2d` sonrası 64×64 görselin 16×16'ya inmesinden gelir.

#### 3. `training_testing_engine.py` — Eğitim Motoru
Modelin öğrenme sürecini yöneten üç fonksiyon içerir:

- **`train_step()`** — Tek bir epoch için eğitim adımı: forward pass → loss hesabı → `zero_grad()` → `backward()` → `optimizer.step()`. Ortalama loss ve accuracy döndürür.
- **`test_step()`** — `torch.inference_mode()` altında (gradyan hesaplamadan) modeli test verisinde değerlendirir.
- **`train()`** — Yukarıdaki iki adımı `epochs` sayısı kadar tekrarlar, her epoch'ta metrikleri ekrana yazdırır ve `results` sözlüğünde (`train_loss`, `train_acc`, `test_loss`, `test_acc`) biriktirir.

#### 4. `utils.py` — Yardımcı Araçlar
- **`save_model()`** — Modelin `state_dict`'ini belirtilen klasöre `.pth` olarak kaydeder (klasör yoksa oluşturur).
- **`get_mean_and_std()`** — Veri setinin kanal bazında ortalama ve standart sapma değerlerini hesaplar. Bu değerler `main.py`'deki `transforms.Normalize()` içinde kullanılır. Modül doğrudan çalıştırıldığında (`__main__`) bu istatistikleri hesaplayıp yazdırır.

#### 5. `main.py` — Orkestrasyon / Giriş Noktası
Tüm parçaları birleştirir ve eğitim sürecini baştan sona çalıştırır:
1. Hiperparametreleri tanımlar.
2. Veri dönüşüm (augmentation + normalizasyon) pipeline'ını kurar.
3. DataLoader'ları oluşturur.
4. Modeli, kayıp fonksiyonunu (`CrossEntropyLoss`) ve optimizeri (`Adam`) tanımlar.
5. `train()` ile modeli eğitir.
6. `save_model()` ile ağırlıkları diske yazar.

#### 6. `load_model_make_prediction.py` — Çıkarım (Inference)
Eğitilmiş modeli üretimde nasıl kullanacağını gösterir:
- Kayıtlı `desert_classifier.pth` ağırlıklarını yükler (`load_state_dict`).
- Tek bir görseli (`data/baklava.jpg`) okur, 64×64'e ölçekler, [0, 1] aralığına normalize eder ve batch boyutu ekler.
- `eval()` + `inference_mode()` altında tahmin yapar, softmax olasılıklarından en yüksek sınıfı seçip ekrana yazar.

---

## ⚙️ Hiperparametreler

`main.py` içinde tanımlıdır:

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `NUM_EPOCHS` | 10 | Eğitim tur sayısı |
| `BATCH_SIZE` | 32 | Batch başına görsel sayısı |
| `HIDDEN_UNITS` | 32 | CNN filtre sayısı |
| `LEARNING_RATE` | 0.001 | Adam öğrenme oranı |

### Veri Ön İşleme (Data Augmentation)
Eğitim sırasında modelin genelleme yeteneğini artırmak için:
- `Resize((64, 64))` — tüm görselleri sabit boyuta getirir.
- `RandomHorizontalFlip(p=0.4)` — %40 olasılıkla yatay çevirme.
- `TrivialAugmentWide()` — otomatik, rastgele augmentation.
- `ToTensor()` — görseli tensöre çevirir.
- `Normalize(mean, std)` — veri setine özel hesaplanmış istatistiklerle normalize eder.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları yükleyin
```bash
pip install -r reqierements.txt
```
> Not: Dosya adı `reqierements.txt` şeklindedir (yazım farkına dikkat).

### 2. Modeli eğitin
```bash
python main.py
```
Eğitim tamamlanınca ağırlıklar `models/desert_classifier.pth` olarak kaydedilir.

### 3. Kayıtlı model ile tahmin yapın
```bash
python load_model_make_prediction.py
```

### 4. (Opsiyonel) Veri seti istatistiklerini hesaplayın
```bash
python utils.py
```

---

## 🔧 Teknik Notlar

- **`torch.multiprocessing.set_start_method('spawn')`** — Windows'ta `num_workers > 0` ile DataLoader'ların doğru çalışması için `main.py`'de ayarlanmıştır.
- Kod şu an **CPU** üzerinde çalışacak şekildedir; `main.py`'deki `device` satırı yorumdadır. GPU kullanmak isterseniz modeli ve tensörleri `.to(device)` ile ilgili cihaza taşımanız gerekir.
- Model ağırlıkları `state_dict` olarak kaydedilir; yüklerken **aynı mimarideki** bir `DesertClassifier` nesnesi oluşturmanız şarttır.

---

## 📚 Bağımlılıklar

- [PyTorch](https://pytorch.org/) (`torch`)
- [TorchVision](https://pytorch.org/vision/) (`torchvision`)

---

## 🎓 Bağlam

Bu proje, **Udemy Deep Learning Bootcamp — Computer Vision** eğitiminin bir parçası olarak, PyTorch'ta modüler ve üretime yakın (production-style) bir proje iskeletinin nasıl kurulacağını gösterir.
