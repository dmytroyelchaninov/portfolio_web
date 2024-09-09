from flask import Flask, request, jsonify, url_for
import os
import sys
from darts.game.image_transform import process_image
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
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False)