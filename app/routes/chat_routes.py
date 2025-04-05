from flask import Blueprint, request, jsonify

from app import db
from app.models import Chat

chat_routes = Blueprint('chat_routes', __name__)

@chat_routes.route('/create', methods=['POST'])
def create_chat() -> tuple:
    user_id = request.form['user_id']

    if not user_id:
        return jsonify({'status': 400, 'message': 'Bad request'}), 400

    db.session.add(Chat(user_id=user_id))
    db.session.commit()

    return jsonify({'status': 200, 'message': 'Chat created successfully'}), 200

@chat_routes.route('/messages/<int:user_id>', methods=['GET'])
def get_chat_messages(user_id: int) -> tuple:
    chat = Chat.query.filter_by(user_id=user_id).first()

    if not chat:
        return jsonify({'status': 404, 'message': 'Chat not found'}), 404

    messages = [message.content for message in chat.messages]

    return jsonify({'status': 200, 'messages': messages}), 200
