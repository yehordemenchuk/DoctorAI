"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import pdfplumber
import docx
from .ocr import get_text_from_pdf

def read_pdf(filename: str, language: str) -> str:
    with pdfplumber.open(filename) as pdf:
        return get_text_from_pdf(pdf, language)

def read_docx(filename: str) -> str:
    return "\n".join(para.text for para in docx.Document(filename).paragraphs)

def read_file(filename: str, language: str) -> str:
    if filename.endswith('docx'):
        return read_docx(filename)

    elif filename.endswith('pdf'):
        return read_pdf(filename, language)

    else:
        return ""