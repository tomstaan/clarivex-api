from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import openai
from werkzeug.utils import secure_filename

# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return "Welcome to Whisper Transcription API!"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join("/tmp", filename)
        file.save(filepath)

        try:
            with open(filepath, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file
                )

            os.remove(filepath)

            return jsonify({"transcription": transcript['text']}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)
