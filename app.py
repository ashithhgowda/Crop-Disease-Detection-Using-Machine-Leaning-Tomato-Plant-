from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Secret key for session management

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained model
model = tf.keras.models.load_model('tomato_disease_model.h5')

# Class names and descriptions
disease_info = {
    'Early Blight': {
        'description': "The dark, irregular spots surrounded by yellowing areas suggest early blight, caused by the fungus Alternaria solani. This condition is common and affects leaves, stems, and fruits.",
        'solution': " Apply fungicides like chlorothalonil or mancozeb, practice crop rotation, and remove infected leaves to improve airflow and reduce the spread of the disease."
    },
    'Healthy': {
        'description': "The leaf appears healthy with no visible signs of disease or damage. This is an indication of good plant health and proper management.",
        'solution': "Continue with good farming practices, including proper watering, sunlight, and pest control."
    },
    'Late Blight': {
        'description': "The presence of dark, water-soaked lesions that may expand and lead to leaf decay indicates late blight, caused by the pathogen Phytophthora infestans. This disease can rapidly destroy crops.",
        'solution': "Use fungicides such as copper-based or metalaxyl solutions to control late blight. Remove and dispose of infected plants immediately, improve air circulation, and avoid overhead watering to minimize moisture on leaves."
    },
    'Tomato___Leaf_Mold': {
        'description': "This disease is characterized by yellow spots on the upper leaf surface and velvety mold growth on the underside. It's caused by the fungus Passalora fulva and thrives in humid conditions.",
        'solution': "Improve ventilation around plants by spacing them properly and reducing humidity through base watering. Remove and dispose of infected leaves to prevent further spread. If necessary, apply fungicides like mancozeb or sulfur for effective control."
    },
    'Tomato___Septoria_leaf_spot': {
        'description': "Small, circular spots with dark margins and light centers suggest Septoria leaf spot, caused by the fungus *Septoria lycopersici*. This disease primarily affects older leaves.",
        'solution': "Prune and discard infected leaves to stop the spread of the fungus. Avoid overhead watering and ensure proper plant spacing to reduce leaf moisture. If needed, apply fungicides like chlorothalonil or copper sulfate for effective disease management."
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': "This condition is caused by tiny spider mites that suck sap from leaves, leading to yellowing and eventual leaf drop. Fine webbing may also be visible.",
        'solution': "Keep plants well-watered and maintain humidity to discourage spider mites. Regularly clean leaves to remove dust and monitor for early signs of infestation. Use insecticidal soap or miticides as needed to control the mites effectively."
    }
}

# Preprocess image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if username and password:
        session['username'] = username
        return redirect(url_for('home'))
    return "Invalid login. Please try again.", 400

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected!", 400
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            processed_image = preprocess_image(file_path)
            predictions = model.predict(processed_image)
            predicted_class = list(disease_info.keys())[np.argmax(predictions[0])]

            detailed_description = disease_info[predicted_class]['description']
            solution = disease_info[predicted_class]['solution']
            os.remove(file_path)

            return render_template('result.html', 
                                   prediction=predicted_class, 
                                   description=detailed_description,
                                   solution=solution)
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
