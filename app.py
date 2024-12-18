from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from werkzeug.utils import secure_filename
import os
from PIL import Image  # Mengimpor Pillow untuk manipulasi gambar

app = Flask(__name__)

# Path ke file model
MODEL_PATH = os.path.join(os.getcwd(), 'training_efficiennet.h5')

# Periksa ketersediaan file model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load model di awal
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

# Kategori hasil prediksi
categories = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Folder untuk upload gambar
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        if file:
            # Simpan file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Menggunakan PIL untuk membuka gambar dan meresize
                img = Image.open(filepath)
                img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model
                img_array = np.array(img)  # Mengonversi gambar ke array numpy
                img_array = img_array / 255.0  # Normalisasi gambar
                img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch

                # Prediksi
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)

                # Validasi hasil prediksi
                if len(predicted_class) > 0 and predicted_class[0] < len(categories):
                    result = categories[predicted_class[0]]
                else:
                    return "Kategori tidak diketahui atau model error.", 500

                # Hasil prediksi
                return f"Hasil Prediksi: {result}"
            except Exception as e:
                # Menambahkan log error lebih rinci
                print(f"Terjadi error saat memproses gambar: {str(e)}")
                return f"Terjadi error saat memproses gambar: {str(e)}", 500

    return '''
        <!doctype html>
        <title>Prediksi Penyakit</title>
        <h1>Unggah Gambar MRI</h1>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Unggah dan Prediksi">
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
