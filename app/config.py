"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""
import os
import secrets

class Config:
    TESSERACT_CMD = r'D:\Tesseract-OCR\tesseract.exe'
    MODEL = "gpt-4o"
    UPLOAD_FOLDER = 'uploads'
    PDF_FOLDER = 'library'
    SECRET_KEY = secrets.token_hex(16)
    TRANSFORMER_EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False