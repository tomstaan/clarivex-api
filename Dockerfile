# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the container
COPY . .

# Expose port 5002 to allow the app to be accessible from outside the container
EXPOSE 5002

# Run the Flask application
CMD ["python", "app.py"]
