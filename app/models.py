from sqlalchemy import Column, String, JSON

from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False)
    login = Column(String(100), unique=True, nullable=False)
    hash_password = Column(String(100), nullable=False)
    chats = db.relationship('Chat', backref='user', lazy=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = Column(String(100), nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    messages = db.relationship('Message', backref='chat', lazy=True)
