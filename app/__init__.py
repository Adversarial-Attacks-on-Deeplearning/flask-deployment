from flask import Flask, render_template, send_from_directory
import os


def create_app():
    app = Flask(__name__, static_folder="../static",
                template_folder="../templates")

    # register each model blueprint
    from .models.unet.routes import unet_bp
    from .models.yolo.routes import yolo_bp
    app.register_blueprint(unet_bp)
    app.register_blueprint(yolo_bp)

    app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, "uploads")

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    @app.route("/", methods=["GET"])
    def home():
        return render_template("home.html")

    return app
