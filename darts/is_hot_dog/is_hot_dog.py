import os
import torch
import warnings
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.realpath(__file__))

model_save_path = os.path.join(current_dir, "is_hot_dog_model.pth")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.classifier = torch.nn.Linear(model.config.hidden_size, 2)

model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def is_hot(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image).logits
        _, predicted = torch.max(outputs, 1)

    return False if predicted.item() == 1 else True

if __name__ == "__main__":
    # image_path = os.path.join(current_dir, "crowns.jpg")
    # result = is_hot(image_path)
    # print(f"That's definetely {result}!")
    pass