"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from flask import Blueprint, Response, request, jsonify
from app.utils.doctor import personal_consultation
from app.utils.file_reader import read_file
from .misc import analyzing_result, is_allowed_filename, save_file, delete_file
import app.utils.ocr as ocr

doctor_routes = Blueprint('doctor_routes', __name__)

@doctor_routes.route('/consult', methods = ['GET'])
def consult() -> Response:
    return jsonify({"advice": personal_consultation(
                    request.form['language'],
                    request.form['specialization'],
                    request.form['question']
                  )
           }
    )

@doctor_routes.route('/analyze', methods = ['GET'])
def analyze_document() -> Response:
    if "file" not in request.files:
        return jsonify({"status": 404, "error": "No file provided"})

    file = request.files['file']

    document_language = request.form['document_language']

    answer_language = request.form['answer_language']

    specialization = request.form['specialization']

    filepath = save_file(file)

    if not file or file.filename == '' or not is_allowed_filename(filepath):
        return jsonify({"status": 404, "error": "Invalid file provided"})

    text = read_file(filepath) if not filepath.endswith(('.jpg', '.jpeg', '.png', '.gif')) \
        else ocr.get_text(document_language, filepath)

    if text == "":
        return jsonify({"status": 404, "error": "Not text extracted"})

    if not delete_file(filepath):
        return jsonify({"status": 404, "error": "File not found"})

    return analyzing_result(document_language, answer_language, specialization, text)