import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2


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


def desenhar_pontos(img_path, coords, saida_path="saida.png"):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Não foi possível carregar a imagem!")

    h, w = img.shape[:2]

    for x_norm, y_norm in coords:
        x = int(x_norm * w)
        y = int(y_norm * h)
        cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(saida_path, img)
    return saida_path
