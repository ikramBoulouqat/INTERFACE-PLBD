import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import glob

# --- config ---
INPUT_SIZE = 256
NUM_CLASSES = 5
ENCODER = "resnet34"

# --- charger le modèle ---
model = smp.Unet(encoder_name=ENCODER, encoder_weights=None, classes=NUM_CLASSES)
model.load_state_dict(torch.load("unet_danger.pth", map_location="cpu"))
model.eval()

preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")

def predict_mask(image_path):
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = preprocess_fn(np.array(img)).astype("float32")
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    mask = torch.argmax(logits, dim=1).squeeze(0).numpy()
    return mask

if __name__ == "__main__":
    fichiers = glob.glob("test1.*")
    if not fichiers:
        print("Aucune image 'test1' trouvée dans le dossier.")
    else:
        print("Image trouvée :", fichiers[0])
        mask = predict_mask(fichiers[0])
        print("masque shape:", mask.shape)
        print("classes présentes:", np.unique(mask))