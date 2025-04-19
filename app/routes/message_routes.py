from flask import Blueprint, jsonify

from app.models import Message
from app.routes.misc import is_user_role_admin, unauthorized_access_message

message_routes = Blueprint('message_routes', __name__)

@message_routes.route("/all")
def get_all_messages() -> tuple:
    if is_user_role_admin():
        return unauthorized_access_message()

    messages = Message.query.all()

    return jsonify({'status': 200,
                    'messages': [{'id': message.id,
                                  'content': message.content,
                                  'chat_id': message.chat.id}
                                 for message in messages]}), 200

@message_routes.route("/<int:message_id>")
def get_message(message_id: int) -> tuple:
    if is_user_role_admin():
        return unauthorized_access_message()

    message = Message.query.get_or_404(message_id)

    return jsonify({'status': 200, 'message':
        {'id': message.id,
         'content': message.content,
         'chat_id': message.chat.id}
        }), 200