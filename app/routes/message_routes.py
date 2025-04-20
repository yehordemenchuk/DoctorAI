from flask import Blueprint, jsonify

from app.models import Message
from app.routes.misc import is_user_role_admin
from flask_jwt_extended.exceptions import NoAuthorizationError

message_routes = Blueprint('message_routes', __name__)

@message_routes.route("/all-messages")
def get_all_messages() -> tuple:
    if not is_user_role_admin():
        raise NoAuthorizationError()

    messages = Message.query.all()

    return jsonify({'status': 200,
                    'messages': [{'id': message.id,
                                  'content': message.content,
                                  'chat_id': message.chat.id}
                                 for message in messages]}), 200

@message_routes.route("/message/<int:message_id>")
def get_message(message_id: int) -> tuple:
    if not is_user_role_admin():
        raise NoAuthorizationError()

    message = Message.query.filter_by(id=message_id).first()

    if not message:
        return jsonify({'status': 404, 'message': 'Message does not exist'}), 404

    return jsonify({'status': 200, 'message':
        {'id': message.id,
         'content': message.content,
         'chat_id': message.chat.id}
        }), 200