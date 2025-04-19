from flask import Blueprint, request, jsonify

from app import db
from app.models import Chat
from app.routes.misc import is_user_role_admin

chat_routes = Blueprint('chat_routes', __name__)

@chat_routes.route('/create', methods=['POST'])
def create_chat() -> tuple:
    user_id = request.form['user_id']

    if not user_id:
        return jsonify({'status': 400, 'message': 'Bad request'}), 400

    db.session.add(Chat(user_id=user_id))
    db.session.commit()

    return jsonify({'status': 200, 'message': 'Chat created successfully'}), 200

@chat_routes.route('/messages/<int:chat_id>', methods=['GET'])
def get_chat_messages(chat_id: int) -> tuple:
    chat = Chat.query.filter_by(chat_id=chat_id).first()

    if not chat:
        return jsonify({'status': 404, 'message': 'Chat not found'}), 404

    messages = [message.content for message in chat.messages]

    return jsonify({'status': 200, 'messages': messages}), 200

@chat_routes.route('/messages', methods=['GET'])
def get_all_chats() -> tuple:
    if is_user_role_admin():
        return jsonify({'status': 403, 'message': 'You are not authorized to access this page'}), 403

    chats = Chat.query.all()

    if not chats:
        return jsonify({'status': 404, 'message': 'Error'}), 404

    return jsonify({'status': 200,
                    'messages': [{'id': chat.id,
                                 'user_id': chat.user_id,
                                 'first_message': chat.messages[0].content}
                                 for chat in chats]}), 200