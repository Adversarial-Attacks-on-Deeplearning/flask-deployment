import albumentations as A
from albumentations.pytorch import ToTensorV2

# Inference (validation) transform—match training val_transforms
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

inference_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[
                1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])
