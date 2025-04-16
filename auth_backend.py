from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secret-key'

CORS(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Simulated user database (we'll use a real DB later)
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

    access_token = create_access_token(identity={"email": email, "store_name": user['store_name']})
    return jsonify(access_token=access_token), 200

@app.route('/user', methods=['GET'])
@jwt_required()
def get_user():
    current_user = get_jwt_identity()
    return jsonify(current_user), 200

if __name__ == '__main__':
    app.run(debug=True)
