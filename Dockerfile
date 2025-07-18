# Use an official Python runtime as a parent image
FROM python:3.12.0-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy only the dependency definitions to leverage Docker cache
COPY pyproject.toml poetry.lock /app/

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --only=main --no-interaction --no-ansi --no-root

# Copy the whisper.cpp directory
COPY whisper.cpp/ /app/whisper.cpp

# Build whisper.cpp
WORKDIR /app/whisper.cpp
RUN rm -rf build && cmake -B build -DGGML_NATIVE=OFF -DGGML_CPU_ARM_FMA=OFF && cmake --build build --config Release

# Return to the main app directory
WORKDIR /app

# Copy the rest of the application's code
COPY iSpeak/ /app/iSpeak

# Command to run the application
CMD ["uvicorn", "iSpeak.api:app", "--host", "0.0.0.0", "--port", "8000"] 