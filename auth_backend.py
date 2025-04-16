from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required,
    get_jwt_identity, get_jwt
)
import os

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'default-secret-for-local-testing')
CORS(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Simulated user database
users = {}

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    store_name = data.get('store_name')

    if email in users:
        return jsonify({"msg": "User already exists"}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    users[email] = {"password": hashed_pw, "store_name": store_name}
    return jsonify({"msg": "Store account created"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = users.get(email)
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({"msg": "Invalid credentials"}), 401

    # Set identity as a string (email) and pass store_name as additional claims
    additional_claims = {"store_name": user['store_name']}
    access_token = create_access_token(identity=email, additional_claims=additional_claims)

    return jsonify(access_token=access_token), 200

@app.route('/user', methods=['GET'])
@jwt_required()
def get_user():
    identity = get_jwt_identity()  # email
    claims = get_jwt()             # includes store_name

    return jsonify({
        "email": identity,
        "store_name": claims.get("store_name")
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render provides PORT
    app.run(host='0.0.0.0', port=port, debug=False)
