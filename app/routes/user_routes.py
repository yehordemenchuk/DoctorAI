from flask import Blueprint, request, jsonify

from app import db
from app.models import User

user_routes = Blueprint('user_routes', __name__)

@user_routes.route('/register', methods=['POST'])
def register() -> tuple:
    username = request.form['username']

    user_login = request.form['login']

    hash_password = request.form['hash_password']

    if not username or not hash_password or not user_login:
        return jsonify({'status': 400, 'message': 'Bad request'}), 400

    if User.query.filter_by(username=username).first() or User.query.filter_by(login=user_login).first():
        return jsonify({'status': 409, 'message': 'Username already exists'}), 409

    db.session.add(User(username=username, login=user_login, hash_password=hash_password))

    db.session.commit()

    return jsonify({'status': 200, 'message': 'Registered successfully'}), 200

@user_routes.route('/login', methods=['GET'])
def login() -> tuple:
    user_login = request.form['login']

    hash_password = request.form['hash_password']

    if not user_login or not hash_password:
        return jsonify({'status': 400, 'message': 'Bad request'}), 400

    user = User.query.filter_by(login=user_login).first()

    if not user:
        return jsonify({'status': 401, 'message': 'User does not exist'}), 401

    if hash_password != user.hash_password:
        return jsonify({'status': 401, 'message': 'Invalid password'}), 401

    return jsonify({'status': 200, 'message': 'Login successfully'}), 200