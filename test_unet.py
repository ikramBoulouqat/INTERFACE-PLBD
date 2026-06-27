import torch
import segmentation_models_pytorch as smp

model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=5)
state = torch.load("unet_danger.pth", map_location="cpu")
print(type(state))
model.load_state_dict(state)
model.eval()
print("OK, modèle charge")