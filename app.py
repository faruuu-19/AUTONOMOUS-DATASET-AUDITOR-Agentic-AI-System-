"""
Hugging Face Spaces entry point.

A Gradio-SDK Space simply runs this file and expects an HTTP server on port
7860, so this launches the existing Flask app (API + built React frontend)
rather than a Gradio interface.

The working directory is switched to backend/ because the strategy and
meta-learning engines persist their state to relative paths
(agent/strategy_memory.pkl, agent/meta_learning.pkl). Running from anywhere
else silently starts them from an empty slate.
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "backend"

sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from api_server import app  # noqa: E402  (import must follow the path setup)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    try:
        from waitress import serve

        serve(app, host="0.0.0.0", port=port, threads=8)
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
