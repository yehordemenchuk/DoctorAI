"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from flask import Flask
from .doctor_routes import doctor_routes

def register_routes(app: Flask):
    app.register_blueprint(doctor_routes)