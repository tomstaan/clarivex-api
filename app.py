from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import openai
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Allowed file extensions for audio files
ALLOWED_EXTENSIONS_AUDIO = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

# Allowed file extensions for images
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'bmp'}

# Load your pre-trained Pixtral-12B model for images using Hugging Face's API
model_path = "/app/checkpoint-30"  # Path to the directory containing model files
model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
model.eval()

# Load processor for handling images
processor = AutoProcessor.from_pretrained(model_path)

# Check if the file is allowed
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return "Welcome to Whisper Transcription and Image Prediction API!"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_AUDIO):
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

# New endpoint for handling image predictions
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected image file"}), 400

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
        filename = secure_filename(file.filename)
        filepath = os.path.join("/tmp", filename)
        file.save(filepath)

        try:
            # Load and process the image
            image = Image.open(filepath)
            inputs = processor(images=image, return_tensors="pt")

            # Perform inference with the model
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

            os.remove(filepath)  # Clean up temporary file
            return jsonify({"prediction": generated_text[0]}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Invalid image format"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)
