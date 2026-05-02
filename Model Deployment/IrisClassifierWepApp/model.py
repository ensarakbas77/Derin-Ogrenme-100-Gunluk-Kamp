"""
model.py
--------
Iris çiçeği sınıflandırması için PyTorch model tanımı.

Mimari:
    Girdi (4 özellik) → Linear(4→16) → ReLU
                      → Linear(16→16) → ReLU
                      → Linear(16→3)  → Çıktı (3 sınıf logit)
"""

import torch.nn as nn


class IrisClassifier(nn.Module):
    """
    Iris veri seti için tam bağlantılı (fully-connected) sinir ağı.

    Katmanlar
    ---------
    - Linear(4, 16)  : 4 özellik → 16 gizli nöron
    - ReLU           : Aktivasyon
    - Linear(16, 16) : Gizli katman
    - ReLU           : Aktivasyon
    - Linear(16, 3)  : 3 sınıf için ham logit çıktısı

    Kullanım
    --------
    >>># model = IrisClassifier()
    >>># model.load_state_dict(torch.load("iris_classification_model.pth"))
    >>># model.eval()
    """

    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4, 16),   # Giriş katmanı: 4 özellik
            nn.ReLU(),          # Aktivasyon
            nn.Linear(16, 16),  # Gizli katman
            nn.ReLU(),          # Aktivasyon
            nn.Linear(16, 3),   # Çıkış katmanı: 3 sınıf
        )

    def forward(self, x):
        """İleri geçiş — ham logit döndürür, softmax uygulamaz."""
        return self.linear_layer_stack(x)
