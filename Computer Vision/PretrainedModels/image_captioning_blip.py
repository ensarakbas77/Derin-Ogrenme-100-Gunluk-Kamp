import sys
import gradio as gr
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Model ve Processor Kurulumu (İlk çalıştırmada otomatik indirilir)
MODEL_NAME = "Salesforce/blip-image-captioning-base"

print("Model yükleniyor, lütfen bekleyin...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()
print("Model başarıyla yüklendi!")


# 2. Tahmin Fonksiyonu
def generate_caption(input_image):
    if input_image is None:
        return "Lütfen bir görsel yükleyin."

    # PIL Image formatına dönüştürme garantisi
    img = Image.fromarray(input_image.astype("uint8"), "RGB")

    # Görseli işle ve tahmin üret
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.capitalize()


# 3. Gradio Arayüz Tasarımı
interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Görsel Yükleyin"),
    outputs=gr.Textbox(label="Modelin Açıklaması (İngilizce)"),
    title="Yapay Zeka Görsel Altyazı Üretici (Image Captioning)",
    description="Hugging Face BLIP modeli kullanarak yüklediğiniz resimleri betimleyin.",
)

# 4. Uygulamayı Başlatma
if __name__ == "__main__":
    interface.launch(share=True)
