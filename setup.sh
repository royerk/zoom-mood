#!/usr/bin/env bash
# Zoom Mood — one-time setup script
set -e

echo "==================================="
echo "  Zoom Mood — Setup"
echo "==================================="
echo ""

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'" 2>/dev/null || {
    echo "❌  Python 3.9 or newer is required."
    echo "   Install it from https://www.python.org/downloads/"
    exit 1
}

echo "✅  Python $(python3 --version | cut -d' ' -f2) found"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦  Creating virtual environment..."
    python3 -m venv .venv
fi

echo "📦  Activating virtual environment..."
source .venv/bin/activate

echo "📦  Installing dependencies (this may take a minute)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "==================================="
echo "  ✅  Setup complete!"
echo "==================================="
echo ""
echo "  To run Zoom Mood:"
echo ""
echo "    source .venv/bin/activate"
echo "    python zoom_mood.py --region"
echo ""
echo "  Then open http://127.0.0.1:5051 in your browser."
echo ""
