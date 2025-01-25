"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from flask import Flask
from flask_cors import CORS

def create_app() -> Flask:
    app = Flask(__name__)

    CORS(app)

    return app