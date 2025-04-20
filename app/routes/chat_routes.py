from flask import Blueprint, request, jsonify

from app import db
from app.models import Chat
from app.routes.misc import is_user_role_admin, get_chats
from flask_jwt_extended.exceptions import NoAuthorizationError

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
    chat = Chat.query.get_or_404(chat_id)

    messages = [message.content for message in chat.messages]

    return jsonify({'status': 200, 'messages': messages}), 200

@chat_routes.route('/user-chats/<int:user_id>', methods=['GET'])
def get_user_chats(user_id: int) -> tuple:
    chats = Chat.query.filter_by(user_id=user_id).all()

    if not chats:
        return jsonify({'status': 404, 'message': 'Error'}), 404

    return get_chats(chats)

@chat_routes.route('/chats', methods=['GET'])
def get_all_chats() -> tuple:
    if not is_user_role_admin():
        raise NoAuthorizationError()

    chats = Chat.query.all()

    if not chats:
        return jsonify({'status': 404, 'message': 'Error'}), 404

    return get_chats(chats)