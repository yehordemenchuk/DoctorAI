"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import g4f
from app.config import Config

def get_response(specialization: str, user_prompt: str) -> str:
    return g4f.ChatCompletion.create(
        model = Config.MODEL,
        messages=[
            {"role": "system", "content": f"""You are a {specialization} specialist, 
                                           who giving personal recommendations to patients"""},
            {"role": "user", "content": user_prompt}
        ]
    )

def personal_consultation(language: str, specialization: str, question: str) -> str:
    return get_response(
        specialization,
        f"Give answer to a question on {language}: {question}"
    )

def analyzing_medical_document(language: str, specialization: str, text: str) -> str:
    return get_response(
        specialization,
        f"Analyze following document and give answer on{language}: {text}"
    )