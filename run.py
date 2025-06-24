from app import create_app
from app.models.unet.model import UNET, DoubleConv


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
