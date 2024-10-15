# Use an official Python runtime as a parent image
FROM python:3.11-slim

# admin can update packages and install AWS command line interface
RUN apt update -y && apt install awscli -y

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies
RUN pip install -r requirements.txt

# Command to run the FastAPI server
CMD ["python3", "app.py"]
