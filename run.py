"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from app import create_app
from app.routes import register_routes

app = create_app()

register_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
