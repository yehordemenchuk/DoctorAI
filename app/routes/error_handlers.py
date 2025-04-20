from flask import jsonify

def unauthorized_access_handler(e):
    return jsonify({'status': 403, 'message': 'You are not authorized to access this page'}), 403

def entity_not_found_handler(e):
    return jsonify({'status': 404, 'message': 'Entity not found'}), 404