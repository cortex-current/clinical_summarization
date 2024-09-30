# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# admin can update packages and install AWS command line interface
RUN apt update -y && apt install awscli -y

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies
RUN pip install -r requirements.txt
# RUN pip install --upgrade accelerate
# RUN pip uninstall -y transformers accelerate
# RUN pip install transformers accelerate

# Command to run the FastAPI server
CMD ["python3", "app.py"]
