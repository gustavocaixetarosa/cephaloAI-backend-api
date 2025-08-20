import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from model import fusionVGG19


class ImagemService:
    def __init__(self, checkpoint_path, device="cuda"):
        # Dispositivo
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Carregar checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        config = checkpoint["config"]  # Namespace salvo no treino
        print("Imprimindo config do checkpoint: " + config)
        self.config = config

        # Criar modelo base (VGG19 sem pesos pré-treinados)
        base_model = models.vgg19(weights=None)

        # Reconstruir modelo customizado
        self.model = fusionVGG19(base_model, config).to(self.device)

        # Carregar pesos treinados
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Criar transform baseado no tamanho usado no treino
        image_h, image_w = config.image_scale
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_h, image_w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5], std=[0.5]
                ),  # ajuste se usou outro valor no treino
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
