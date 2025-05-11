#!/bin/bash
# Simple script to run just the API without Docker for testing

echo "=========================================="
echo "   Water Potability API Server"
echo "=========================================="
echo

# Check for virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if ! pip show fastapi &>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p logs monitoring

# Start the API server
echo "Starting FastAPI server..."
echo "Access the web interface at http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

uvicorn src.scripts.deploy_api:app --host 0.0.0.0 --port 8000 --reload
