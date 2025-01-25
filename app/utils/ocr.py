"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import pytesseract
from PIL import Image
from app.config import Config

def get_text(image_path: str) -> str:
    pytesseract.pytesseract.tesseract_cmd =  Config.TESSERACT_CMD

    image = Image.open(image_path)

    return pytesseract.image_to_string(image)