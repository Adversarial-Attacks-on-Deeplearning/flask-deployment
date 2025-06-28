# routes.py additions for YOLO object detection
from flask import Blueprint, request, render_template, current_app
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from . import attacks

# If you want custom preprocessing:
# ensure this returns a tensor or Path-based processing
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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Load YOLO model once on import
yolo_model = YOLO("yolov8n_TrafficSigns.pt")
yolo_model.to(device)

# Load a default font for labels
font = ImageFont.truetype("DejaVuSans.ttf", size=28)

attack_funcs = {
    'dis_dag': lambda img: attacks.disappearance_dag_attack(img, model_path=yolo_model, num_iterations=20,
                                                            gamma=0.03,
                                                            conf_threshold=0.25,
                                                            device='cpu'
                                                            ),
    'target_dag': lambda img: attacks.targeted_dag_attack(
        img,
        model_path=yolo_model,
        adversarial_class=0,
        num_iterations=20,
        gamma=0.003,
        conf_threshold=0.25,
        device='cpu'),
    'fool': lambda img: attacks.fool_detectors_attack(
        img,
        model_path=yolo_model,
        num_iterations=20,
        gamma=0.01,
        conf_threshold=0.25,
        lambda_reg=0.01,
        device='cpu'
    ),
    'fgsm': lambda img: attacks.fgsm_attack_detector(
        img,
        model=yolo_model,
        epsilon=0.08,
        conf_threshold=0.25,
        device='cpu'
    ),
    'uap': lambda img: attacks.uap(img, device='cpu')



}


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


@yolo_bp.route('/', methods=['GET', 'POST'])
def index():
    device = 'cpu'
    orig_image = det_image = adv_image = None
    attack_applied = False
    selected_attack = None

    if request.method == 'POST':
        f = request.files.get('file')
        selected_attack = request.form.get('attack', '')
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(path)
            orig_image = filename

            # Preprocess to tensor
            try:
                tensor = preprocess_image(path).to(device)
            except Exception:
                tensor = None

            # If an attack is selected, apply it
            if tensor is not None and selected_attack in attack_funcs:
                adv_tensor = attack_funcs[selected_attack](path)
                # Convert back to PIL
                adv_np = (adv_tensor.squeeze().permute(
                    1, 2, 0).cpu().numpy() * 255).astype('uint8')
                adv_pil = Image.fromarray(adv_np)
                adv_name = f"adv_{selected_attack}_{filename}"
                adv_pil.save(os.path.join(UPLOAD_FOLDER, adv_name))
                attack_applied = True
                adv_image = adv_name
                input_file = adv_name
            else:
                input_file = filename

            # Run YOLO detection
            results = yolo_model(os.path.join(
                UPLOAD_FOLDER, input_file), device=device)
            det = results[0]

            # Draw boxes on input_file
            img = Image.open(os.path.join(
                UPLOAD_FOLDER, input_file)).convert('RGB')
            draw = ImageDraw.Draw(img)
            for box, conf, cls in zip(det.boxes.xyxy, det.boxes.conf, det.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = f"{yolo_model.names[int(cls)]}: {conf:.2f}"
                # Yellow box
                draw.rectangle([x1, y1, x2, y2], outline='yellow', width=5)
                # Label background
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2]-tb[0], tb[3]-tb[1]
                draw.rectangle([x1, y1-th-6, x1+tw+6, y1], fill='yellow')
                # White text
                draw.text((x1+3, y1-th-4), label, fill='black', font=font)

            det_image = f"det_{input_file}"
            img.save(os.path.join(UPLOAD_FOLDER, det_image))

    return render_template(
        'yolo/index.html',
        orig_image=orig_image,
        det_image=det_image,
        adv_image=adv_image,
        attack_applied=attack_applied,
        selected_attack=selected_attack
    )
