"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

from app import create_app, db
from app.routes import register_routes
from flask_migrate import Migrate

app = create_app()
migrate = Migrate(app, db)

register_routes(app)

if __name__ == '__main__':
    app.run(debug=True)
