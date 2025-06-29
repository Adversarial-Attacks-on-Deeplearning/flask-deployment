import os
import numpy as np
import tensorflow as tf
from flask import Blueprint, request, render_template, current_app
from werkzeug.utils import secure_filename
from PIL import Image

# utilities and class mapping
from .GTSRB_utils import predict_traffic_sign, GTSRB_CLASSES
# adversarial attack functions for classification
from .attacks import fgsm_attack_single_image, pgd_attack_single_image, mi_fgsm, jsma_attack

# Blueprint
efficient_bp = Blueprint(
    "efficientnet",
    __name__,
    url_prefix="/models/efficientnet",
)

# Config
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

# Allowed file check


def allowed_file(fname):
    return "." in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT


# Load classifier
MODEL_FILE = os.path.join(os.path.dirname(__file__), "EfficientNetB1.keras")
classifier = tf.keras.models.load_model(MODEL_FILE)
classifier.trainable = False

# Load defense model
DEFENSE_FILE = os.path.join(os.path.dirname(
    __file__), "efficientnet_adv_epoch_21.keras")
defense_model = tf.keras.models.load_model(DEFENSE_FILE)
defense_model.trainable = False

attack_funcs = {
    'fgsm': lambda img, lbl: fgsm_attack_single_image(model=classifier, image=img, label=tf.convert_to_tensor(lbl, dtype=tf.int32), epsilon=0.03, normalized=False),
    'mi-fgsm': lambda img, lbl: mi_fgsm(model=classifier, x=img, y=tf.convert_to_tensor(lbl, dtype=tf.int32), epsilon=0.05, T=3, mu=1.0),
    'pgd': lambda img, lbl: pgd_attack_single_image(model=classifier, image=img, label=tf.convert_to_tensor(lbl, dtype=tf.int32), epsilon=0.2, alpha=0.04, iterations=3, normalized=False),
    'jsma': lambda img, lbl: jsma_attack(model=classifier, image=img, target_label=1, gamma=0.1, theta=255, num_pixels=50)
}


@efficient_bp.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    orig_label = orig_conf = None
    pert_label = pert_conf = None
    def_label = def_conf = None
    selected_attack = selected_defense = None
    pert_name = def_name = None
    attack_applied = defense_applied = False

    if request.method == 'POST':
        f = request.files.get('file')
        selected_attack = request.form.get('attack')
        selected_defense = request.form.get('defense')
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            f.save(path)

            # preprocess
            img = Image.open(path).convert('RGB').resize((240, 240))
            x = np.expand_dims(np.array(img), axis=0)

            # original prediction
            preds = classifier.predict(x)
            orig_id = predict_traffic_sign(x, classifier, GTSRB_CLASSES)
            orig_label = GTSRB_CLASSES[orig_id]
            orig_conf = float(np.max(preds))

            # apply attack if requested
            if selected_attack in attack_funcs:
                attack_applied = True
                adv_x = attack_funcs[selected_attack](
                    x if selected_attack != 'jsma' else img, orig_id)
                adv_arr = adv_x.numpy() if hasattr(adv_x, 'numpy') else adv_x
                adv_arr = np.reshape(adv_arr, adv_arr.shape[-3:])
                adv_x = np.expand_dims(adv_arr, axis=0)
                adv_preds = classifier.predict(adv_x)
                pert_id = int(np.argmax(adv_preds))
                pert_label = GTSRB_CLASSES[pert_id]
                pert_conf = float(np.max(adv_preds))
                arr = (
                    adv_x[0] * 255).astype(np.uint8) if adv_x.max() <= 1.0 else adv_x[0].astype(np.uint8)
                adv_img = Image.fromarray(arr)
                pert_name = f"pert_{selected_attack}_{filename}"
                adv_img.save(os.path.join(
                    current_app.config['UPLOAD_FOLDER'], pert_name))

            # apply defense if requested on adversarial example
            if attack_applied and selected_defense == 'defend':
                defense_applied = True
                # choose input for defense: perturbed or original
                defend_input = adv_x if attack_applied else x
                def_preds = defense_model.predict(defend_input)
                def_id = int(np.argmax(def_preds))
                def_label = GTSRB_CLASSES[def_id]
                def_conf = float(np.max(def_preds))
                def_name = f"def_{selected_attack}_{filename}"
                # reuse saved pert image or original to show
                source_arr = (defend_input[0] * 255).astype(
                    np.uint8) if defend_input.max() <= 1.0 else defend_input[0].astype(np.uint8)
                def_img = Image.fromarray(source_arr)
                def_img.save(os.path.join(
                    current_app.config['UPLOAD_FOLDER'], def_name))

    return render_template(
        'efficientnet/index.html',
        filename=filename,
        orig_label=orig_label,
        orig_conf=orig_conf,
        pert_label=pert_label,
        pert_conf=pert_conf,
        def_label=def_label,
        def_conf=def_conf,
        pert_image=pert_name,
        defense_image=def_name,
        attack_applied=attack_applied,
        defense_applied=defense_applied,
        selected_attack=selected_attack,
        selected_defense=selected_defense
    )
