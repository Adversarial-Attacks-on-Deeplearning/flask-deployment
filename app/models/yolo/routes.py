# routes.py additions for YOLO object detection with defense
import os
from flask import Blueprint, request, render_template, current_app
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from . import attacks
from .utils import preprocess_image

# Blueprint setup
yolo_bp = Blueprint(
    "yolo",
    __name__,
    url_prefix="/models/yolo",
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


# Device
device = 'cpu'

# Load YOLO models
yolo_model = YOLO("yolov8n_TrafficSigns.pt").to(device)
defense_model = YOLO(
    "yolov8n_TrafficSigns_adversarial_training_v2.pt").to(device)

# Font for labels
font = ImageFont.truetype("DejaVuSans.ttf", size=28)

attack_funcs = {
    'dis_dag': lambda img: attacks.disappearance_dag_attack(img, model_path=yolo_model, num_iterations=20, gamma=0.03, conf_threshold=0.25, device=device),
    'target_dag': lambda img: attacks.targeted_dag_attack(img, model_path=yolo_model, adversarial_class=0, num_iterations=20, gamma=0.003, conf_threshold=0.25, device=device),
    'fool': lambda img: attacks.fool_detectors_attack(img, model_path=yolo_model, num_iterations=20, gamma=0.01, conf_threshold=0.25, lambda_reg=0.01, device=device),
    'fgsm': lambda img: attacks.fgsm_attack_detector(img, model=yolo_model, epsilon=0.08, conf_threshold=0.25, device=device),
    'uap': lambda img: attacks.uap(img, device=device)
}


@yolo_bp.route('/', methods=['GET', 'POST'])
def index():
    orig_image = det_image = adv_image = def_image = None
    attack_applied = defense_applied = False
    selected_attack = selected_defense = None

    if request.method == 'POST':
        f = request.files.get('file')
        selected_attack = request.form.get('attack', '')
        selected_defense = request.form.get('defense', '')
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(path)
            orig_image = filename

            # Choose input image for detection
            input_file = filename

            # Apply attack if requested
            if selected_attack in attack_funcs:
                adv_tensor = attack_funcs[selected_attack](path)
                adv_np = (adv_tensor.squeeze().permute(
                    1, 2, 0).cpu().numpy() * 255).astype('uint8')
                adv_pil = Image.fromarray(adv_np)
                adv_image = f"adv_{selected_attack}_{filename}"
                adv_pil.save(os.path.join(UPLOAD_FOLDER, adv_image))
                attack_applied = True
                input_file = adv_image

            # Apply defense if requested on adversarial image
            if attack_applied and selected_defense == 'defend':
                # run defense model detection
                results_def = defense_model(os.path.join(
                    UPLOAD_FOLDER, input_file), device=device)
                det_def = results_def[0]
                img_def = Image.open(os.path.join(
                    UPLOAD_FOLDER, input_file)).convert('RGB')
                draw_def = ImageDraw.Draw(img_def)
                for box, conf, cls in zip(det_def.boxes.xyxy, det_def.boxes.conf, det_def.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label = f"{defense_model.names[int(cls)]}: {conf:.2f}"
                    draw_def.rectangle(
                        [x1, y1, x2, y2], outline='blue', width=5)
                    tb = draw_def.textbbox((0, 0), label, font=font)
                    tw, th = tb[2]-tb[0], tb[3]-tb[1]
                    draw_def.rectangle([x1, y1-th-6, x1+tw+6, y1], fill='blue')
                    draw_def.text((x1+3, y1-th-4), label,
                                  fill='white', font=font)
                def_image = f"def_{filename}"
                img_def.save(os.path.join(UPLOAD_FOLDER, def_image))
                defense_applied = True
                det_image = def_image
            else:
                # Standard detection on (adv or orig)
                results = yolo_model(os.path.join(
                    UPLOAD_FOLDER, input_file), device=device)
                det = results[0]
                img = Image.open(os.path.join(
                    UPLOAD_FOLDER, input_file)).convert('RGB')
                draw = ImageDraw.Draw(img)
                for box, conf, cls in zip(det.boxes.xyxy, det.boxes.conf, det.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label = f"{yolo_model.names[int(cls)]}: {conf:.2f}"
                    draw.rectangle([x1, y1, x2, y2], outline='yellow', width=5)
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2]-tb[0], tb[3]-tb[1]
                    draw.rectangle([x1, y1-th-6, x1+tw+6, y1], fill='yellow')
                    draw.text((x1+3, y1-th-4), label, fill='black', font=font)
                det_image = f"det_{input_file}"
                img.save(os.path.join(UPLOAD_FOLDER, det_image))

    return render_template(
        'yolo/index.html',
        orig_image=orig_image,
        adv_image=adv_image,
        det_image=det_image,
        attack_applied=attack_applied,
        defense_applied=defense_applied,
        selected_attack=selected_attack,
        selected_defense=selected_defense
    )
