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

### Production with Nginx reverse proxy (recommended)

```
# The app runs on HTTP locally, Nginx handles HTTPS
poetry run uvicorn iSpeak.api:app --host 127.0.0.1 --port 8000
```

### Direct HTTPS (alternative, if not using Nginx)

```
poetry run uvicorn iSpeak.api:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem
```

## Download Whisper.cpp
