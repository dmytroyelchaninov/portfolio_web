from flask import Flask, request, jsonify, url_for, render_template
from flask_mail import Mail, Message
import os
import sys
from darts.game.image_transform import process_image
from is_hot_dog.is_hot_dog import is_hot
import uuid


script_dir = os.path.dirname(os.path.abspath(__file__))
yolos_dir = os.path.join(script_dir, 'yolos')
sys.path.append(yolos_dir)

app = Flask(__name__, static_folder=os.path.join(script_dir, '../static'))

UPLOAD_FOLDER = os.path.join(script_dir, '../static/uploads/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024     # 15MB

@app.route('/process_image', methods=['POST'])
def game():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    print('Received request')
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = f"{str(uuid.uuid4())}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except Exception as e:
                return jsonify({'message': f'Failed to save file: {str(e)}'}), 500
            try:
                print(f'Processing image: {filepath}')
                processed_image_path, score_list = process_image(filepath)
                score = ', '.join(map(str, score_list))
            except Exception as e:
                return jsonify({'message': f'Error processing image: {str(e)}'}), 500
            print(f'Processed image: {processed_image_path}')    
            processed_image_url = url_for('static', filename=f'uploads/{os.path.basename(processed_image_path)}')
            print(f'Processed image URL: {processed_image_url}')
            return jsonify({
                'image_url': processed_image_url,
                'score': score
            })
            
        else:
            return jsonify({'message': 'Invalid file type'}), 400

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'dmytroyelchaninov@gmail.com'
app.config['MAIL_PASSWORD'] = 'wpte dell andx dpni'

mail = Mail(app)

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    message_content = data.get('message')

    msg = Message(subject='Someone sent you message from website',
                  sender=app.config['MAIL_USERNAME'],
                  recipients=['dmytroyelchaninov@gmail.com'],
                  body=message_content)
    try:
        mail.send(msg)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'status': 'failed', 'reason': str(e)}), 500
    

@app.route('/hot_dog', methods=['GET', 'POST'])
def hot_dog():
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    print('Received request')
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = f"{str(uuid.uuid4())}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except Exception as e:
                return jsonify({'message': f'Failed to save file: {str(e)}'}), 500
            try:
                print(f'Checking if it is a hot dog: {filepath}')
                result = is_hot(filepath)
                print(f'Is it a hot dog: {result}')
            except Exception as e:
                return jsonify({'message': f'Error checking if it is a hot dog: {str(e)}'}), 500
            return jsonify({
                'result': result
            })
            
        else:
            return jsonify({'message': 'Invalid file type'}), 400
    
    # return render_template('hot_dog.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)
    # app.run(debug=True, port=5000)