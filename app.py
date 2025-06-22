import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F

from model import DoubleConv, UNET
from transforms import inference_transform, IMAGE_HEIGHT, IMAGE_WIDTH

# Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "unet_state_dict.pth"
UAP_PATH = 'uap_perturbation.pth'

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = torch.load(
    'unet_complete_model.pth',
    map_location=device,
    weights_only=False
).to(device)
model.eval()

# Load UAP perturbation
uap = torch.load(UAP_PATH, map_location=device)
delta = uap['delta'].to(device)
print("Delta max:", delta.abs().max().item())
epsilon = uap['epsilon']

# Helpers


def preprocess(img: Image.Image):
    arr = np.array(img.convert('RGB'), dtype=np.uint8)
    aug = inference_transform(image=arr)
    return aug['image'].unsqueeze(0).to(device)


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_mask(inp):
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


def predict_mask_general(img: Image.Image, perturb_func=None):
    original_size = img.size
    inp = preprocess(img)
    if perturb_func is not None:
        pert = perturb_func(inp)
    else:
        pert = inp
    print("Input min:", inp.min().item(), "Input max:", inp.max().item())
    print("Perturbed min:", pert.min().item(),
          "Perturbed max:", pert.max().item())
    mask_array = compute_mask(pert)
    small_mask = Image.fromarray(mask_array)
    mask_resized = small_mask.resize(original_size, resample=Image.NEAREST)

    # Convert pert to PIL and resize to original size
    pert_pil = tensor_to_pil(pert)
    pert_resized = pert_pil.resize(original_size, resample=Image.NEAREST)

    used_img = pert_resized if perturb_func is not None else img
    return mask_resized, used_img

# Perturbation functions


def uap_attack(inp):
    scaled_delta = 0.25 * delta
    return torch.clamp(inp + scaled_delta, 0, 1)


def deepfool_attack(inp, max_iter=30, overshoot=0.02):
    target_mask = (torch.sigmoid(model(inp)) > 0.5).float()
    inp = inp.clone().detach().to(device)
    target_mask = target_mask.clone().detach().to(device)

    pert_inp = inp.clone().detach().requires_grad_(True)
    r_total = torch.zeros_like(inp).to(device)
    loop_i = 0

    with torch.enable_grad():
        while loop_i < max_iter:
            pert_inp.requires_grad = True
            output = model(pert_inp)
            pred = (torch.sigmoid(output) > 0.5).float()

            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, target_mask)
            grad = torch.autograd.grad(loss, pert_inp, retain_graph=False)[0]

            w = grad / (grad.norm() + 1e-8)
            r_i = (loss + 1e-4) * w

            r_total = (r_total + r_i).clamp(-overshoot, overshoot)
            pert_inp = torch.clamp(inp + r_total, 0, 1).detach()

            loop_i += 1

    return pert_inp.detach()


attack_funcs = {
    'uap': uap_attack,
    'deepfool': deepfool_attack,
    # 'another': another_attack,
}


@app.route('/models/unet', methods=['GET', 'POST'])
def index():
    orig, clean_mask = None, None
    if request.method == 'POST':
        f = request.files.get('file')
        if f and allowed_file(f.filename):
            name = secure_filename(f.filename)
            path = os.path.join(UPLOAD_FOLDER, name)
            f.save(path)
            img = Image.open(path).convert('RGB')
            mask, _ = predict_mask_general(img, perturb_func=None)
            clean_mask = 'mask_clean_' + name
            mask.save(os.path.join(UPLOAD_FOLDER, clean_mask))
            orig = name
    return render_template('index.html', orig_image=orig, clean_mask=clean_mask)


@app.route('/attack', methods=['POST'])
def attack():
    orig = request.form['orig']
    attack_type = request.form['attack']
    path = os.path.join(UPLOAD_FOLDER, orig)
    img = Image.open(path).convert('RGB')

    pert_mask, pert_img = None, None
    if attack_type in attack_funcs:
        pert_mask, pert_img = predict_mask_general(
            img, perturb_func=attack_funcs[attack_type])
        pm_name = f'mask_{attack_type}_' + orig
        im_name = f'pert_{attack_type}_' + orig
        pert_mask.save(os.path.join(UPLOAD_FOLDER, pm_name))
        pert_img.save(os.path.join(UPLOAD_FOLDER, im_name))
        return render_template('index.html',
                               orig_image=orig,
                               clean_mask=request.form.get('clean_mask'),
                               pert_image=im_name,
                               pert_mask=pm_name,
                               attack_applied=True)
    return render_template('index.html',
                           orig_image=orig,
                           clean_mask=request.form.get('clean_mask'),
                           attack_applied=False)


@app.route('/static/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
