# Whisper Transcription API

This is a Flask-based API that transcribes audio files using OpenAI's Whisper model.

## Features

- Transcribes audio files into text.
- Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm.
- Uses OpenAI Whisper API for transcription.

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/whisper-transcription-api.git
   cd whisper-transcription-api
Create a virtual environment and activate it:

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up your OpenAI API key in a .env file:

Create a .env file in the root directory with the following contents:

makefile
Copy code
OPENAI_API_KEY=sk-proj-your_openai_key
Run the Flask application:

bash
Copy code
python app.py
Docker Setup
To run the service using Docker, follow these steps:

Build the Docker image:

bash
Copy code
docker build -t whisper-backend .
Run the Docker container:

bash
Copy code
docker run -d -p 5002:5002 --env-file .env whisper-backend
Testing the API
To test the transcription service, you can use curl:

bash
Copy code
curl -X POST http://localhost:5002/transcribe -F "file=@/path_to_your_audio_file.mp3"
Make sure to replace /path_to_your_audio_file.mp3 with the actual path of your audio file.

API Endpoints
GET /
Returns a welcome message to confirm the API is running.

POST /transcribe
Uploads an audio file and returns a transcription.

Request Body: multipart/form-data

file: The audio file to be transcribed.
Response:

Success: JSON with the transcription.
Error: JSON with an error message.
Notes
The audio file size is limited to 25 MB.
Supported audio formats are mp3, mp4, mpeg, mpga, m4a, wav, webm.
Dependencies
Flask
OpenAI API
Python-dotenv
Werkzeug
markdown
Copy code

### 6. **Key Fixes & Points**
- **File Extensions:** Make sure the audio file you upload is in one of the supported formats (`mp3`, `mp4`, etc.).
- **Temporary File:** You are saving the uploaded file to a temporary directory (`/tmp`). Make sure your Docker environment allows writing to this directory.
- **OpenAI API Key:** Ensure the `.env` file is correctly set with your OpenAI API key.

Once all these changes are made, you should be able to transcribe your audio files correctly! Let me know if you need further adjustments!