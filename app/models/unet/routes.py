import os
from flask import Blueprint, request, render_template, current_app
from werkzeug.utils import secure_filename
from PIL import Image
import torch

from .model import UNET, DoubleConv
from .transforms import inference_transform, IMAGE_HEIGHT, IMAGE_WIDTH
from .attack import uap_attack, deepfool_attack, FGSM_single_image, MI_fgsm_single_image, PGD_single_image
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
DEFENSE_PATH = os.path.join(MODULE_DIR, "my_checkpoint_adv_2.pth.tar")

device = 'cpu'

# original model
model = UNET(in_channels=3, out_channels=1).to(device)
model = torch.load(MODEL_PATH,  map_location=device,
                   weights_only=False).to(device)
model.eval()

# defense model
defense_model = UNET(in_channels=3, out_channels=1).to(device)
defense_ckpt = torch.load(DEFENSE_PATH, map_location=device)
defense_model.load_state_dict(defense_ckpt.get('state_dict', defense_ckpt))
defense_model.eval()

# UAP
uap = torch.load(UAP_PATH, map_location=device)
delta = uap["delta"].to(device)
epsilon = uap.get("epsilon", None)

attack_funcs = {
    "uap": uap_attack,
    "deepfool": deepfool_attack,
    'fgsm': FGSM_single_image,
    'mi-fgsm': MI_fgsm_single_image,
    'pgd': PGD_single_image
}


def allowed_file(fname):
    return "." in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@unet_bp.route("/", methods=["GET", "POST"])
def index():
    upload_dir = current_app.config['UPLOAD_FOLDER']
    orig = None
    clean_mask = None
    pert_image = None
    pert_mask = None
    defense_mask = None
    attack_applied = False
    defense_applied = False

    if request.method == "POST":
        f = request.files.get("file")
        selected_attack = request.form.get("attack", None)
        selected_defense = request.form.get("defense", None)
        if f and allowed_file(f.filename):
            name = secure_filename(f.filename)
            os.makedirs(upload_dir, exist_ok=True)
            path = os.path.join(upload_dir, name)
            f.save(path)
            img = Image.open(path).convert("RGB")

            # original segmentation
            mask, _ = predict_mask_general(
                img,
                None,
                model,
                inference_transform,
                delta
            )
            clean_mask = f"mask_clean_{name}"
            mask.save(os.path.join(upload_dir, clean_mask))
            orig = name

            # apply attack if selected
            if selected_attack in attack_funcs:
                attack_applied = True
                pert_mask, pert_img = predict_mask_general(
                    img,
                    attack_funcs[selected_attack],
                    model,
                    delta,
                    device
                )
                pm_name = f"mask_{selected_attack}_{orig}"
                im_name = f"pert_{selected_attack}_{orig}"
                pert_mask.save(os.path.join(upload_dir, pm_name))
                pert_img.save(os.path.join(upload_dir, im_name))
                pert_mask = pm_name
                pert_image = im_name

            # apply defense if selected and attack was applied
            if attack_applied and selected_defense == 'defend':
                defense_applied = True
                # run defense_model on perturbed image
                pert_path = os.path.join(upload_dir, pert_image)
                pert_pil = Image.open(pert_path).convert("RGB")
                def_mask, _ = predict_mask_general(
                    pert_pil,
                    None,
                    defense_model,
                    inference_transform,
                    delta
                )
                def_name = f"mask_defended_{orig}"
                def_mask.save(os.path.join(upload_dir, def_name))
                defense_mask = def_name

    return render_template(
        "unet/index.html",
        orig_image=orig,
        clean_mask=clean_mask,
        pert_image=pert_image,
        pert_mask=pert_mask,
        defense_mask=defense_mask,
        attack_applied=attack_applied,
        defense_applied=defense_applied
    )
