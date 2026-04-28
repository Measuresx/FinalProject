from transformers import ViTForImageClassification, AutoImageProcessor
from PIL import Image
import torch
import os
print("Current working dir:", os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained("./PokemonModel").to(device)
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

img = Image.open("test.jpg").convert("RGB")

inputs = processor(images=img, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    predicted_id = outputs.logits.argmax(-1).item()

print(model.config.id2label[predicted_id])