import numpy as np
from PIL import Image
import torch
from .transforms import inference_transform, IMAGE_HEIGHT, IMAGE_WIDTH


def preprocess(img, device):
    arr = np.array(img.convert('RGB'), dtype=np.uint8)
    aug = inference_transform(image=arr)
    return aug['image'].unsqueeze(0).to(device)


def compute_mask(inp, model):
    with torch.no_grad():
        probs = torch.sigmoid(model(inp))
    mask = (probs.squeeze() >= 0.5).cpu().numpy().astype(np.uint8) * 255
    return mask


def tensor_to_pil(tensor):
    # Convert tensor [0,1] range, shape (1, C, H, W) to PIL Image
    tensor = tensor.squeeze(0).cpu()  # (C, H, W)
    tensor = tensor.permute(1, 2, 0)  # (H, W, C)
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def predict_mask_general(img, perturb_func, model, delta, device):
    original_size = img.size
    inp = preprocess(img, device)
    if perturb_func is not None:
        pert = perturb_func(inp, device=device, model=model, delta=delta)
    else:
        pert = inp
    print("Input min:", inp.min().item(), "Input max:", inp.max().item())
    print("Perturbed min:", pert.min().item(),
          "Perturbed max:", pert.max().item())
    mask_array = compute_mask(pert, model)
    small_mask = Image.fromarray(mask_array)
    mask_resized = small_mask.resize(original_size, resample=Image.NEAREST)

    # Convert pert to PIL and resize to original size
    pert_pil = tensor_to_pil(pert)
    pert_resized = pert_pil.resize(original_size, resample=Image.NEAREST)

    used_img = pert_resized if perturb_func is not None else img
    return mask_resized, used_img
