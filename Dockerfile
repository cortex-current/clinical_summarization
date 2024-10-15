# Use an official Python runtime as a parent image
FROM python:3.11-slim

# admin can update packages and install AWS command line interface
RUN apt update -y && apt install awscli -y

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for the application
COPY app.py /app/
# COPY main.py /app/
# COPY config.yaml /app/
# COPY params.yaml /app/
COPY pyproject.toml /app/
COPY .github/workflows /app/
COPY requirements.txt /app/
# COPY clinical_summary /app/clinical_summary
COPY templates /app/templates
COPY static /app/static

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the dependencies
RUN pip install -r requirements.txt

# Command to run the FastAPI server
CMD ["python3", "app.py"]
