import torch
from torch import nn

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Modeli eğitim moduna alıyoruz
    model.train()

    # Train loss ve train accuracy değerlerini başlatıyoruz
    train_loss, train_acc = 0, 0

    # DataLoader içindeki batch'ler üzerinde döngü kuruyoruz
    for batch, (X, y) in enumerate(dataloader):

        # 1. Forward pass
        y_pred = model(X)

        # 2. Loss değerini hesapla ve toplam train loss'a ekle
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer gradientlerini sıfırla
        optimizer.zero_grad()

        # 4. Backpropagation işlemini yap
        loss.backward()

        # 5. Optimizer ile model parametrelerini güncelle
        optimizer.step()

        # Batch bazında accuracy hesapla ve toplam train accuracy değerine ekle
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Ortalama train loss ve train accuracy değerlerini hesapla
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Modeli değerlendirme moduna alıyoruz
    model.eval()

    # Test loss ve test accuracy değerlerini başlatıyoruz
    test_loss, test_acc = 0, 0

    # Test aşamasında gradient hesaplamaya gerek yoktur
    with torch.inference_mode():
        # DataLoader içindeki batch'ler üzerinde döngü kuruyoruz
        for batch, (X, y) in enumerate(dataloader):
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Loss değerini hesapla ve toplam test loss'a ekle
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Batch bazında accuracy hesapla ve toplam test accuracy değerine ekle
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Ortalama test loss ve test accuracy değerlerini hesapla
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5) -> object:
    # Sonuçları saklamak için boş bir dictionary oluşturuyoruz
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Belirlenen epoch sayısı kadar eğitim ve test adımlarını çalıştırıyoruz
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        # Eğitim sürecinde neler olduğunu ekrana yazdırıyoruz
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Sonuçları dictionary içine ekliyoruz
        # Eğer değer tensor formatındaysa CPU'ya alıp float değere çeviriyoruz
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # Epoch'lar bittikten sonra tüm sonuçları döndürüyoruz
    return results