# Configuration file for iSpeak backend
from pathlib import Path

# Get the absolute path to the backend directory
BACKEND_DIR = Path(__file__).parent.parent.absolute()

# Define paths relative to the backend directory
WHISPER_PATH = BACKEND_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli"
MODEL_PATH = BACKEND_DIR / "whisper.cpp" / "models" / "ggml-base.en.bin"
PRAAT_SCRIPT_PATH = BACKEND_DIR / "scripts" / "formant_analysis.praat"

# Validate paths exist
def validate_paths():
    """Validate that required paths exist."""
    if not WHISPER_PATH.exists():
        raise FileNotFoundError(
            f"Whisper CLI not found at {WHISPER_PATH}. "
            "Please ensure whisper.cpp is installed in the backend directory."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Whisper model not found at {MODEL_PATH}. "
            "Please download the model file to the correct location."
        )
