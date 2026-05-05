import os
import torch
import numpy as np
from PIL import Image
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    ViTForImageClassification,
    ViTImageProcessor
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# LOAD DETECTION MODEL (find Pokémon in big image)
# -------------------------------------------------
print("Loading detector...")
det_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
det_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

# -------------------------------------------------
# LOAD CLASSIFIER (your model)
# -------------------------------------------------
print("Loading Pokémon classifier...")
clf_id = "skshmjn/Pokemon-classifier-gen9-1025"
clf_model = ViTForImageClassification.from_pretrained(clf_id).to(device)
clf_processor = ViTImageProcessor.from_pretrained(clf_id)

# -------------------------------------------------
# STEP 1 — Detect objects in image
# -------------------------------------------------
def detect_objects(image_pil, threshold=0.8):
    inputs = det_processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = det_model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]]).to(device)
    results = det_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    return results["boxes"].cpu().numpy()

# -------------------------------------------------
# STEP 2 — Crop objects and save
# -------------------------------------------------
def crop_objects(image_pil, boxes, save_dir="crops"):
    os.makedirs(save_dir, exist_ok=True)
    crops = []

    img_np = np.array(image_pil)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img_np[y1:y2, x1:x2]

        crop_pil = Image.fromarray(crop)
        save_path = f"{save_dir}/pokemon_{i}.png"
        crop_pil.save(save_path)

        crops.append((save_path, crop_pil))

    return crops

# -------------------------------------------------
# STEP 3 — Classify one crop
# -------------------------------------------------
def classify_crop(image_pil):
    inputs = clf_processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = clf_model(**inputs)

    pred_id = outputs.logits.argmax(-1).item()
    name = clf_model.config.id2label[pred_id]

    return pred_id + 1, name

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def run_pipeline(image_path):
    print("\nProcessing:", image_path)

    image = Image.open(image_path).convert("RGB")

    # Detect Pokémon
    boxes = detect_objects(image)
    print(f"Detected {len(boxes)} objects")

    if len(boxes) == 0:
        print("No Pokémon detected.")
        return

    # Crop them
    crops = crop_objects(image, boxes)

    # Classify each crop (THIS IS YOUR SECOND PROGRAM RUNNING HERE)
    print("\n===== FINAL RESULTS =====")
    for path, crop_img in crops:
        dex, name = classify_crop(crop_img)
        print(f"{path} → {name} (Pokédex #{dex})")

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    run_pipeline("img3.jpg")
