"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import os
from flask import jsonify, Response
from werkzeug.datastructures import FileStorage
#from werkzeug.utils import secure_filename

from app import db
from app.models import Message
from app.utils.doctor import analyzing_medical_document
from app.config import Config

def analyzing_result(document_language: str,
                     answer_language: str, specialization: str, text: str) -> Response:
    return jsonify({"status": 200, "analyze": analyzing_medical_document(document_language, answer_language,
                                                                         specialization, text)})

def is_allowed_filename(filename: str) -> bool:
    filename_parts = filename.rsplit('.')

    return filename_parts[len(filename_parts) - 1].lower() in ('jpg', 'jpeg', 'png', 'gif', 'docx', 'pdf')

def save_file(file: FileStorage) -> str:
    if not os.path.exists(Config.UPLOAD_FOLDER):
        os.makedirs(Config.UPLOAD_FOLDER)

    filepath = os.path.join(Config.UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    return filepath

def delete_file(filepath: str) -> bool:
    if os.path.exists(filepath):
        os.remove(filepath)

        return True

    return False

def save_message(content: str, chat_id: str):
    db.session.add(Message(content=content, chat_id=int(chat_id)))
    db.session.commit()