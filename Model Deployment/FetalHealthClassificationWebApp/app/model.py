"""
model.py
--------
Fetal sağlık sınıflandırması için PyTorch model mimarisi.
Eğitim sırasında kullanılan mimariyle birebir aynı olmalıdır.
"""

import torch
import torch.nn as nn


class ClassificationModel(nn.Module):
    """
    3 sınıflı fetal sağlık sınıflandırıcı.
    Giriş  : 11 KTG özelliği
    Çıkış  : 3 logit (Normal / Şüpheli / Patolojik)
    """

    def __init__(self):
        super().__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(11, 20),
            nn.ReLU(),

            nn.Linear(20, 20),
            nn.ReLU(),

            nn.Linear(20, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)
