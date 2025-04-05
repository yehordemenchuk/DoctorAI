from sqlalchemy import Column, Integer, String, Boolean, Text, ForeignKey, text

from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    login = Column(String(100), unique=True, nullable=False)
    hash_password = Column(String(100), nullable=False)
