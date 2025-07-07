import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from core.networks import SeemoRe
from accelerate import Accelerator
import torchvision.transforms as transforms

lq_image_path = './dir'
model_path = './dir'

accelerator = Accelerator()
device = accelerator.device

model = SeemoRe().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

image = Image.open(lq_image_path).convert('RGB')
transform = transforms.ToTensor()
lq_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    sr_tensor = model(lq_tensor)

lq_image = ToPILImage()(lq_tensor.squeeze(0).cpu().clamp(0, 1))
sr_image = ToPILImage()(sr_tensor.squeeze(0).cpu().clamp(0, 1))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(lq_image)
plt.title('lq')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sr_image)
plt.title('sr')
plt.axis('off')

plt.tight_layout()
plt.show()