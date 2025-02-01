"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import g4f
from app.config import Config

def get_response(specialization: str, user_prompt: str) -> str:
    return g4f.ChatCompletion.create(
        model = Config.MODEL,
        messages = [
            {"role": "system", "content": f"Real professional {specialization} doctor"},
            {"role": "user", "content": f"""Act as real, qualified and professional doctor 
                                        specialized on {specialization}.""" + user_prompt}
        ]
    )

def personal_consultation(language: str, specialization: str, question: str) -> str:
    return get_response(
        specialization,
        f"""Provide clear, accurate, and ethical advice strictly on {language} advices, 
        ensuring grammatical correctness, remove words not in answering language or invalid chars, 
        to the patient's question {question}."""
    )

def analyzing_medical_document(document_language: str,
                               answer_language: str, specialization: str, text: str) -> str:
    return get_response(
        specialization,
        f"""Analyze following medical document on {document_language} 
        and give answer on {answer_language} and before returning an answer 
        validate answer grammar, spelling and remove invalid chars, that can`t be read by human, and 
        words not in answer language, and you can`t answer "I can`t help you" 
        if answer is by your specialization if all is correct, 
        just return an answer, don`t write about it`s corectness: {text}."""
    )