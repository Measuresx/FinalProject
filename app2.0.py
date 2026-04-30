from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and image processor
model_id = "skshmjn/Pokemon-classifier-gen9-1025"
model = ViTForImageClassification.from_pretrained(model_id).to(device)
image_processor = ViTImageProcessor.from_pretrained(model_id)

# Load and process an image
img = Image.open('test.jpg').convert("RGB")
inputs = image_processor(images=img, return_tensors='pt').to(device)

# Make predictions
outputs = model(**inputs)
predicted_id = outputs.logits.argmax(-1).item()
predicted_pokemon = model.config.id2label[predicted_id]

# Print predicted class
print(f"Predicted Pokémon Pokédex number: {predicted_id+1}")
print(f"Predicted Pokémon: {predicted_pokemon}")
