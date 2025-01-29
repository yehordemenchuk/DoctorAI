"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import pytesseract
from PIL import Image
from app.config import Config
from pdfplumber.pdf import PDF

def initialize_pytesseract():
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD

def get_text_from_image(language: str, image_path: str) -> str:
    initialize_pytesseract()

    image = Image.open(image_path)

    return pytesseract.image_to_string(image, lang=language)

def get_text_from_pdf(pdf: PDF, language: str) -> str:
    initialize_pytesseract()

    return "\n".join(pytesseract.image_to_string(page.to_image().original, lang=language) for page in pdf.pages)