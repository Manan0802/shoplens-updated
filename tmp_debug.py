import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
img=np.random.randint(0,255,(224,224,3),dtype=np.uint8)
image=Image.fromarray(img)
inputs=processor(images=image, return_tensors='pt')
out=model.get_image_features(**inputs)
print(type(out))
print(out.shape if hasattr(out,'shape') else None)
print('fields', dir(out) if not isinstance(out, torch.Tensor) else 'tensor')
