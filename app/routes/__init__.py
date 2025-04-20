"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from flask import Flask

from .chat_routes import chat_routes
from .doctor_routes import doctor_routes
from .error_handlers import unauthorized_access_handler
from .message_routes import message_routes
from .user_routes import user_routes
from flask_jwt_extended.exceptions import NoAuthorizationError


def register_error_handlers():
    for bp in [message_routes, user_routes, chat_routes]:
        bp.register_error_handler(NoAuthorizationError, unauthorized_access_handler)

def register_routes(app: Flask):
    register_error_handlers()

    app.register_blueprint(doctor_routes)
    app.register_blueprint(user_routes)
    app.register_blueprint(chat_routes)
    app.register_blueprint(message_routes)