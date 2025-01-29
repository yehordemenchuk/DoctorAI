"""© 2025 DoctorAI. All rights reserved. Use of this application constitutes acceptance of the Privacy Policy and Terms of Use.
DoctorAI is not a medical institution and does not replace professional medical consultation."""

import g4f
from app.config import Config

def get_response(specialization: str, user_prompt: str) -> str:
    return g4f.ChatCompletion.create(
        model = Config.MODEL,
        messages = [
            {"role": "system", "content": f"Real professional {specialization} doctor"},
            {"role": "user", "content": f"""If you were a real qualified doctor, specialized 
                                        on {specialization}""" + user_prompt}
        ]
    )

def personal_consultation(language: str, specialization: str, question: str) -> str:
    return get_response(
        specialization,
        f"""What clear, accurate, and ethical advice strictly in {language} advices did you provide, 
        ensuring grammatical correctness, words not in answering language or incorrect symbols, 
        to the patient's question {question}."""
    )

def analyzing_medical_document(document_language: str,
                               answer_language: str, specialization: str, text: str) -> str:
    return get_response(
        specialization,
        f"""Analyze following medical document on {document_language} 
        and give answer on {answer_language} and before answering 
        validate grammar, spelling and remove invalid chars, if all is
        correct, just return an answer: {text}"""
    )