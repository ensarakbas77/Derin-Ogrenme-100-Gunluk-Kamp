# YOLO Webcam Object Detection

Webcam üzerinden gerçek zamanlı nesne tespiti yapan basit bir Python uygulaması. [Ultralytics YOLO](https://docs.ultralytics.com/) modeli kullanılarak kameradan gelen görüntü akışı işlenir; tespit edilen nesneler hem ekrana bounding box olarak çizilir hem de konsola yazdırılır.

## Özellikler

- Gerçek zamanlı webcam görüntüsü
- YOLO modeli ile nesne tespiti (COCO veri seti — 80 sınıf)
- Bounding box ve etiketlerin canlı çizimi
- Tespit edilen nesnelerin (yalnızca değişiklik olduğunda) konsola yazdırılması
- Yapılandırılabilir model, kamera indeksi ve güven eşiği

## Gereksinimler

- Python 3.9+
- Bağlı bir webcam

## Kurulum

```bash
git clone https://github.com/<kullanici-adi>/<repo-adi>.git
cd <repo-adi>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Kullanım

```bash
python object_detection_yolo.py
```

İlk çalıştırmada YOLO model dosyası (`yolo11n.pt`) otomatik olarak indirilir.

- Uygulamadan çıkmak için OpenCV penceresi seçiliyken **`q`** tuşuna basın.

## Yapılandırma

`object_detection_yolo.py` dosyasının başındaki sabitleri değiştirerek davranışı özelleştirebilirsiniz:

| Sabit | Açıklama | Varsayılan |
|---|---|---|
| `MODEL_NAME` | Kullanılacak YOLO model dosyası | `"yolo11n.pt"` |
| `CAMERA_INDEX` | Webcam indeksi (birden fazla kamera için) | `0` |
| `CONFIDENCE_THRESHOLD` | Minimum güven skoru (0.0 – 1.0) | `0.5` |

Daha doğru fakat daha yavaş tespit için aşağıdaki model varyantlarını deneyebilirsiniz:

| Model | Boyut | Hız | Doğruluk |
|---|---|---|---|
| `yolo11n.pt` | nano | en hızlı | en düşük |
| `yolo11s.pt` | small | hızlı | düşük |
| `yolo11m.pt` | medium | orta | orta |
| `yolo11l.pt` | large | yavaş | yüksek |
| `yolo11x.pt` | xlarge | en yavaş | en yüksek |

## Proje Yapısı

```
.
├── object_detection_yolo.py    # Ana uygulama
├── requirements.txt            # Python bağımlılıkları
├── .gitignore
└── README.md
```

## Lisans

MIT
