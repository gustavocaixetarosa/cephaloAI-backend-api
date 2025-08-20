import torch
import torchvision.transforms as transforms
from PIL import Image


class ImagemService:
    def __init__(self, checkpoint_path, device="cuda"):
        # Dispositivo
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Carregar modelo salvo como objeto
        self.model = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        ).to(self.device)

        self.model.eval()

        # Definir transform (ajuste image_h, image_w para o que você usou no treino)
        image_h, image_w = 200, 160  # <<< substitua pelo seu image_scale
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_h, image_w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5], std=[0.5]
                ),  # ajuste se usou outra normalização
            ]
        )

    def predict(self, image_path):
        # Abrir imagem
        image = Image.open(image_path).convert("RGB")

        # Aplicar transform
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)

        return outputs
