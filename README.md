## Install Coolify

https://coolify.io/docs/get-started/installation
SSh and Firewall guide

### Add PostgreSQL ressource

postgresql v17

## Run backend server locally

### Development (HTTP only)

```
poetry run uvicorn iSpeak.api:app --reload --port 8000
```

### Production with Nginx reverse proxy

```
# The app runs on HTTP locally, Nginx handles HTTPS
poetry run uvicorn iSpeak.api:app --reload --host 0.0.0.0 --port 8000
```

## Download Whisper.cpp

Follow this repo [Whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/README.md)
