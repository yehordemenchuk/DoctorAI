"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from flask import Blueprint, request, jsonify
from app.utils.doctor import personal_consultation
from app.utils.file_reader import read_file
from .misc import analyzing_result, is_allowed_filename, save_file, delete_file, save_message
import app.utils.ocr as ocr

doctor_routes = Blueprint('doctor_routes', __name__)

@doctor_routes.route('/consult', methods = ['GET'])
def consult() -> tuple:
    question = request.form['question']

    chat_id = request.form['chat_id']

    if not question or not chat_id:
        return jsonify({"status": 400, "error": "Bad Request"}), 400

    save_message(question, chat_id)

    advice = personal_consultation(
        request.form['language'],
        request.form['specialization'],
        question
    )

    save_message(advice, chat_id)

    return jsonify({"advice": advice}), 200

@doctor_routes.route('/analyze', methods = ['GET'])
def analyze_document() -> tuple:
    if "file" not in request.files:
        return jsonify({"status": 400, "error": "No file provided"}), 400

    file = request.files['file']

    document_language = request.form['document_language']

    answer_language = request.form['answer_language']

    specialization = request.form['specialization']

    chat_id = request.form['chat_id']

    if not chat_id:
        return jsonify({"status": 400, "error": "Bad Request"}), 400

    filepath = save_file(file)

    save_message(filepath, chat_id)

    if not file or file.filename == '' or not is_allowed_filename(filepath):
        return jsonify({"status": 400, "error": "Invalid file provided"}), 400

    text = read_file(filepath, document_language) if filepath.endswith(('.pdf', '.docx')) \
        else ocr.get_text_from_image(document_language, filepath)

    if text == "":
        return jsonify({"status": 404, "error": "Not text extracted"}), 400

    if not delete_file(filepath):
        return jsonify({"status": 404, "error": "File not found"}), 404

    return analyzing_result(document_language, answer_language, specialization, text), 200