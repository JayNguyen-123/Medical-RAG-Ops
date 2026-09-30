import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Safe defaults for offline unit tests; real values (if present) take precedence.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-dummy")
os.environ.setdefault("MEDICAL_ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("ALLOWED_CLEARANCE_GROUPS", "cardiology,oncology")
os.environ.setdefault("TRACING_ENABLED", "false")
