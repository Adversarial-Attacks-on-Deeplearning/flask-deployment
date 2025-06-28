# routes.py additions for YOLO object detection
from flask import Blueprint, request, render_template, current_app
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLO model once on import
yolo_model = YOLO("yolov8n_TrafficSigns.pt")
yolo_model.to(device)

# Load a default font for labels
font = ImageFont.truetype("DejaVuSans.ttf", size=30)


def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


@yolo_bp.route("/", methods=["GET", "POST"])
def index():
    orig_image, det_image = None, None
    if request.method == "POST":
        f = request.files.get("file")
        if f and allowed_file(f.filename):
            # Secure and save uploaded file
            filename = secure_filename(f.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            f.save(filepath)
            orig_image = filename

            # Attempt custom preprocess inference
            try:
                img_tensor = preprocess_image(filepath)
                results = yolo_model.predict(img_tensor, device=device)
            except Exception:
                results = yolo_model(filepath, device=device)

            det = results[0]

            # Annotate image
            img = Image.open(filepath).convert("RGB")
            draw = ImageDraw.Draw(img)
            for box, score, cls in zip(det.boxes.xyxy, det.boxes.conf, det.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = f"{yolo_model.names[int(cls)]}: {score:.2f}"

                # Yellow bounding box
                draw.rectangle([x1, y1, x2, y2], outline="yellow", width=10)

                # Measure text via textbbox
                tb = draw.textbbox((0, 0), label, font=font)
                text_width, text_height = tb[2] - tb[0], tb[3] - tb[1]

                # Background rectangle for text
                bg = [
                    x1,                    # left
                    y1 - text_height - 4,  # top
                    x1 + text_width + 4,   # right
                    y1                     # bottom
                ]
                draw.rectangle(bg, fill="yellow")

                # black text over yellow
                text_pos = (x1 + 2, y1 - text_height - 2)
                draw.text(text_pos, label, fill="black", font=font)

            # Save annotated image
            det_image = f"det_{filename}"
            img.save(os.path.join(UPLOAD_FOLDER, det_image))

    return render_template(
        "yolo/index.html",
        orig_image=orig_image,
        det_image=det_image,
    )
