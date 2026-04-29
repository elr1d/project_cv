from flask import Flask, request, render_template, url_for, send_from_directory, jsonify, session, Response
from pathlib import Path
from app.DATABASE.DB_FUNC import check_duplicate, save_files_to_db_and_folder_transactioned,get_unused_count
from app.DATABASE.DB_CREATE import init_db
from app.train.fine_tune import uploaded_model_tune
from app.website.model_manager import ModelManager
import uuid
import hashlib
import time
import threading
import os
import secrets
app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / 'images'
IMAGE_FOLDER = BASE_DIR.parent / 'data' / 'uploaded_train'
EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
CONTENT_VOLUME = 10 * 1024 * 1024
model_manager = ModelManager()
app.config['MAX_CONTENT_LENGTH'] = CONTENT_VOLUME
AMOUNT_TO_RETRAIN = 100
model_manager.load_last_model()
current_model_info = model_manager.get_model_info()
info_lock = threading.Lock()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Нет файла в запросе'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    if not check_extension(file.filename):
        return jsonify({'error': 'Некорректное расширение файла'}), 400
    
    old_path = session.get('user_image')
    if old_path:
        old_path = Path(old_path)
        if old_path.exists():
            old_path.unlink()
    
    extension = file.filename.rsplit('.', 1)[1].lower()
    file_name = f'{uuid.uuid4().hex}.{extension}'
    file.save(Path(UPLOAD_FOLDER) / file_name)
    
    session['user_image'] = str(UPLOAD_FOLDER / file_name)
    return jsonify({'url': url_for('uploaded_file', filename=file_name),
                    'filename': file_name})
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/determine', methods=['POST'])
def determine():
    image_path = session.get('user_image')
    if not image_path or not Path(image_path).exists():
        return jsonify({'error' : 'Файл изображения не найден'}),500
    try:
        predicted_class, confidence = model_manager.predict(image_path)
        return jsonify({'result': f'Это { 'Собака' if predicted_class else 'Кот'}, \nуверенность: {int(confidence*100)}%'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def check_extension(filename):
    return filename.rsplit('.', 1)[1].lower() in EXTENSIONS


@app.route('/server/upload', methods=['POST'])
def upload_to_server():
    
    if 'images' not in request.files:
        return jsonify({'error': 'Нет файлов в запросе'}), 400
    
    to_save = []
    duplicates = []
    files = request.files.getlist('images')
    
    if not files:
        return jsonify({'error': 'Файлы не выбраны'}), 400
    
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400

        file_data = file.read()
        if not file_data:
            return jsonify({'error': 'Файл пустой'}), 400

        file_hash = compute_file_hash(file_data)

        if check_duplicate(file_hash):
            duplicates.append(file.filename)
            continue
        
        path_names = file.filename.replace('\\', '/').split('/')
        if len(path_names) > 1:
            class_name = path_names[-2]
            original_name = path_names[-1]
        else:
            original_name = path_names[0]
            if not original_name.startswith('cat') and not original_name.startswith('dog'):
                return jsonify({'error': 'Некорректное имя файла'}), 400
            class_name = 'cats' if original_name.startswith('cat') else 'dogs'

        if not check_extension(file.filename):
            return jsonify({'error': 'Некорректное расширение файла'}), 400

        if class_name.lower() not in ('cats', 'dogs'):
            return jsonify({'error': 'Некорректный класс'}), 400

        extension = original_name.rsplit('.', 1)[1].lower()
        unique_name = f'{uuid.uuid4().hex}.{extension}'
        file_path = Path(IMAGE_FOLDER) / class_name / unique_name

        to_save.append((file_data, str(file_path), class_name, file_hash))

    saved_count = 0
    class_count = None
    if to_save:
        saved_count, class_count = save_files_to_db_and_folder_transactioned(to_save)
        if isinstance(saved_count, tuple):
            return jsonify(saved_count), class_count
    if get_unused_count() >= AMOUNT_TO_RETRAIN:
        threading.Thread(
            target=uploaded_model_tune_and_update,
            args=(model_manager.model_path,),   
            daemon=True
        ).start()
    return jsonify({
        'saved': saved_count,
        'duplicates': len(duplicates),
        'class': class_count
    })

def uploaded_model_tune_and_update(checkpoint_path):
    result = uploaded_model_tune(checkpoint_path)
    if result:
        change_model()
def compute_file_hash(data):
    return hashlib.sha256(data).hexdigest()

def update_model_info():
    global current_model_info
    with info_lock:
        current_model_info = model_manager.get_model_info()

def change_model():
    model_manager.load_last_model()
    update_model_info()
    
@app.route('/stream')
def stream():
    def generate():
        last_sent = None
        while True:
            while True:
                with info_lock:
                    current = current_model_info
                if current != last_sent:
                    break
                time.sleep(0.5)
            yield f"data: {current}\n\n"
            last_sent = current
    
    return Response(generate(), mimetype="text/event-stream")

if __name__ == '__main__':
    
    init_db()
    app.run(debug=True)
    
