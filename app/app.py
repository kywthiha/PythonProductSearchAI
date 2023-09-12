import os

from flask import Flask, render_template, request, flash, send_from_directory
from werkzeug.utils import secure_filename

from feature_extraction import FeatureExtractor
from image_indexer import ImageIndexer
from image_search_engine import ImageSearchEngine

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

# Initialize feature extractor and image indexer
resnet_feature_extractor = FeatureExtractor('ResNet50')
index_name = 'flask_test_image_search_v1'
product_images_folder_path = 'uploads'

image_indexer = ImageIndexer(index_name, product_images_folder_path, resnet_feature_extractor)
image_indexer.create_index()

search_engine = ImageSearchEngine(index_name, resnet_feature_extractor)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        uploaded_files = request.files.getlist('user_images[]')

        for user_image in uploaded_files:
            if user_image and allowed_file(user_image.filename):
                # Save the uploaded image to the uploads folder
                filename = secure_filename(user_image.filename)
                user_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # Index the uploaded image
                image_indexer.index_single_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            flash('Image uploaded and indexed successfully.')

    return render_template('upload.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Handle image search
        user_image = request.files['user_image']
        if user_image and allowed_file(user_image.filename):
            # Save the uploaded image to a temporary location
            filename = secure_filename(user_image.filename)
            user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            user_image.save(user_image_path)

            # Perform similarity search
            search_results = search_engine.perform_similarity_search(user_image_path)
            return render_template('index.html', search_results=search_results)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
