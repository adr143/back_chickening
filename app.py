import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import sounddevice as sd
import librosa
import time
from tensorflow.keras.models import load_model


# ----------------- Load Models ------------------
# YOLOv8 model
model = YOLO("Chicken_Cory.pt")

# Audio classification model
audio_model = load_model("Chicken_Audio.keras")

# ----------------- Flask Setup ------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///track_record.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOW_UNSAFE_WORKER'] = True

socketio = SocketIO(app, cors_allowed_origins="*")

db = SQLAlchemy(app)

video = cv2.VideoCapture(0)

coryza_detected = 0
total_chickens_detected = 0
CATEGORIES = ["Healthy", "Unhealthy", "Noise"]
sr = 22050
duration = 2
threshold = 0.6

class ChickenDiagnosis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    day = db.Column(db.Integer, nullable=False)
    time = db.Column(db.String(20), nullable=False)
    num_chickens = db.Column(db.Integer, nullable=False)
    num_infected = db.Column(db.Integer, nullable=False)
    diagnosis_result = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<ChickenDiagnosis {self.date} - {self.time}>"
    
def extract_features_from_audio(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec)
    features = np.mean(log_mel_spec, axis=1)
    return features

def generate_video():
    global coryza_detected, total_chickens_detected

    while True:
        success, frame = video.read()
        if not success:
            break

        try:
            results = model.predict(frame, conf=0.5)
            frame = results[0].plot()
        except Exception as e:
            print("Error during prediction or plotting:", e)
            continue


        total_count = 0
        coryza_count = 0

        for result in results[0].boxes.data:
            total_count += 1  # every detection = one chicken

            class_id = int(result[-1])
            class_name = model.names[class_id]
            if class_name.lower() == "coryza":
                coryza_count += 1

        total_chickens_detected = total_count
        coryza_detected = coryza_count

        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_stream', {
            'image': frame_encoded,
            'coryza_detected': coryza_detected,
            'total_detected': total_chickens_detected
        })
        time.sleep(0.1)
        
def generate_audio_classification():
    while True:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()

        y = audio.flatten()
        if np.abs(y).mean() < 0.01:
            continue

        if not np.isfinite(y).all():
            print("Non-finite audio detected. Skipping...")
            continue

        # Normalize (optional)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        features = extract_features_from_audio(y, sr=sr)
        features = features.reshape(features.shape[0], 1)
        features = np.expand_dims(features, axis=0)

        prediction = audio_model.predict(features, verbose=0)
        confidence = float(np.max(prediction))
        predicted_index = int(np.argmax(prediction))

        label = CATEGORIES[predicted_index]

        socketio.emit("audio_classification", {
            "label": label,
            "confidence": confidence,
            "waveform": y.tolist()
        })
        
        time.sleep(0.1)

@socketio.on('connect')
def connect():
    socketio.start_background_task(generate_video)
    # socketio.start_background_task(generate_audio_classification)
    print("Client connected")


@socketio.on('disconnect')
def disconnect():
    print("Client disconnected")
    
@app.route('/record', methods=['POST'])
def record_diagnosis():
    from datetime import datetime
    global total_chickens_detected, coryza_detected

    try:
        now = datetime.now()
        date = now.date()
        day = now.isoweekday()
        time_str = now.strftime("%H:%M")

        num_chickens = total_chickens_detected
        num_infected = coryza_detected
        diagnosis_result = "Disease Detected" if num_infected > 0 else "No Disease Detected"

        new_record = ChickenDiagnosis(
            date=date,
            day=day,
            time=time_str,
            num_chickens=num_chickens,
            num_infected=num_infected,
            diagnosis_result=diagnosis_result
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            "message": "Record saved",
            "data": {
                "date": str(date),
                "day": day,
                "time": time_str,
                "num_chickens": num_chickens,
                "num_infected": num_infected,
                "diagnosis_result": diagnosis_result
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/diagnosis', methods=['GET'])
def get_diagnosis_records():
    try:
        # Fetch all records from the ChickenDiagnosis table
        records = ChickenDiagnosis.query.all()
        
        # Convert to a list of dictionaries
        diagnosis_data = []
        for record in records:
            diagnosis_data.append({
                "id": record.id,
                "date": str(record.date),
                "day": record.day,
                "time": record.time,
                "num_chickens": record.num_chickens,
                "num_infected": record.num_infected,
                "diagnosis_result": record.diagnosis_result
            })

        return jsonify({
            "message": "Records retrieved successfully",
            "data": diagnosis_data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, )
