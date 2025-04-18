"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from flask import Flask

from .chat_routes import chat_routes
from .doctor_routes import doctor_routes
from .user_routes import user_routes


def register_routes(app: Flask):
    app.register_blueprint(doctor_routes)
    app.register_blueprint(user_routes)
    app.register_blueprint(chat_routes)