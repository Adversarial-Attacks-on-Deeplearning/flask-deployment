import os
from flask import Blueprint, request, render_template, current_app, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch

from .model import UNET, DoubleConv
from .transforms import inference_transform, IMAGE_HEIGHT, IMAGE_WIDTH
from .attack import uap_attack, deepfool_attack
from .utils import preprocess, compute_mask, tensor_to_pil, predict_mask_general


unet_bp = Blueprint(
    "unet",
    __name__,
    url_prefix="/models/unet",
)

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# load model & UAP once on import
MODULE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODULE_DIR, "unet_complete_model.pth")

UAP_PATH = os.path.join(MODULE_DIR, "uap_perturbation.pth")

device = 'cpu'

model = UNET(in_channels=3, out_channels=1).to(device)

model = torch.load(MODEL_PATH,  map_location=device,
                   weights_only=False).to(device)
model.eval()

uap = torch.load(UAP_PATH, map_location=device)
delta = uap["delta"].to(device)
epsilon = uap["epsilon"]

attack_funcs = {
    "uap": uap_attack,
    "deepfool": deepfool_attack,
}


def allowed_file(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@unet_bp.route("/", methods=["GET", "POST"])
def index():
    upload_dir = current_app.config['UPLOAD_FOLDER']
    orig, clean_mask = None, None
    if request.method == "POST":
        f = request.files.get("file")
        if f and allowed_file(f.filename):
            name = secure_filename(f.filename)
            os.makedirs(upload_dir, exist_ok=True)
            path = os.path.join(upload_dir, name)
            f.save(path)
            img = Image.open(path).convert("RGB")
            mask, _ = predict_mask_general(
                img, None, model, inference_transform, delta)
            clean_mask = f"mask_clean_{name}"
            mask.save(os.path.join(upload_dir, clean_mask))
            orig = name
    return render_template("unet/index.html", orig_image=orig, clean_mask=clean_mask)


@unet_bp.route("/attack", methods=["POST"])
def attack():
    orig = request.form["orig"]
    attack_type = request.form["attack"]
    upload_dir = current_app.config['UPLOAD_FOLDER']
    path = os.path.join(upload_dir, orig)
    img = Image.open(path).convert("RGB")

    if attack_type in attack_funcs:
        pert_mask, pert_img = predict_mask_general(
            img,
            attack_funcs[attack_type],
            model,
            delta,
            device
        )
        pm_name = f"mask_{attack_type}_{orig}"
        im_name = f"pert_{attack_type}_{orig}"
        pert_mask.save(os.path.join(upload_dir, pm_name))
        pert_img.save(os.path.join(upload_dir, im_name))
        return render_template(
            "unet/index.html",
            orig_image=orig,
            clean_mask=request.form.get("clean_mask"),
            pert_image=im_name,
            pert_mask=pm_name,
            attack_applied=True,
        )
    return render_template(
        "unet/index.html",
        orig_image=orig,
        clean_mask=request.form.get("clean_mask"),
        attack_applied=False,
    )
