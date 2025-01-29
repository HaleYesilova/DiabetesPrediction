from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Kaydedilmiş modeli yükle
model = joblib.load("diabet.pkl")


# Ana sayfa rotası
@app.route('/home')
def home():
    return render_template('index.html')  # Bir form içeren HTML dosyasını oluşturun


# Tahmin rotası
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Formdan gelen verileri al
        input_features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        # Modeli kullanarak tahmin yap
        prediction = model.predict([input_features])

        # Tahmini metin olarak düzenle
        output = "Diabet hastası" if prediction[0] == 0 else "Diabet hastası değil"

        return render_template('index.html', prediction_text=f"Sonuç: {output}")

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)