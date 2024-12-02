from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.transcription import transcribe_audio
import os


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

#CORS(app, resources={r"/transcribe": {"origins": "http://localhost:5174"}})


UPLOAD_FOLDER = "./recordings"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
       # file_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
        file_path = os.path.join("./recordings/audio.wav")
        file.save(file_path)  # Save the uploaded file
        transcription = transcribe_audio(file_path)
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        print(f"Error occurred during transcription: {e}")
        return jsonify({"error": "Failed to transcribe audio"}), 500

if __name__ == "__main__":
    app.run(debug=True)
