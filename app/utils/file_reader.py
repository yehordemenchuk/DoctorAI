"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import pdfplumber
import docx

def read_pdf(filename: str) -> str:
    with pdfplumber.open(filename) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def read_docx(filename: str) -> str:
    return "\n".join([para.text for para in docx.Document(filename).paragraphs])

def read_file(filename: str) -> str:
    if filename.endswith('docx'):
        return read_docx(filename)

    elif filename.endswith('pdf'):
        return read_pdf(filename)

    else:
        return ""