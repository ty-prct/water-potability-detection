#!/bin/bash
# Run the complete MLOps pipeline for testing

set -e  # Exit on any error

# Display header
echo "========================================"
echo "   Water Potability MLOps Pipeline Test"
echo "========================================"

# Update permissions
chmod +x setup.sh

# Setup environment
echo -e "\n[1/7] Setting up environment..."
./setup.sh

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    pip install -q -r requirements.txt --no-cache-dir
fi

# Run training pipeline
echo -e "\n[2/7] Running training pipeline..."
python src/scripts/train_pipeline.py

# Run evaluation
echo -e "\n[3/7] Evaluating models..."
python src/scripts/evaluate_pipeline.py

# Run tests
echo -e "\n[4/7] Running tests..."
pytest src/tests/

# Check Docker status and build Docker image
echo -e "\n[5/7] Checking Docker status and building Docker image..."
echo "Checking Docker service..."
if systemctl is-active --quiet docker; then
    echo "Docker service is running"
else
    echo "Docker service is not running, attempting to start..."
    sudo systemctl start docker || true
    sleep 2
fi

# Check if Docker is accessible
if docker info &>/dev/null; then
    echo "Docker is accessible, building image..."
    docker build -t water-potability:test .
else
    echo "WARNING: Docker is not accessible. You may need to:"
    echo "1. Ensure Docker is installed"
    echo "2. Ensure Docker service is running: sudo systemctl start docker"
    echo "3. Add your user to docker group: sudo usermod -aG docker \$USER"
    echo "4. Log out and log back in for group changes to take effect"
    echo "5. Or run the ./docker_troubleshoot.sh script for assistance"
    echo "Skipping Docker build step..."
fi

# Run model monitoring (in background)
echo -e "\n[6/7] Starting model monitoring..."
python src/scripts/model_monitoring.py &
MONITORING_PID=$!

# Start API server and test it
echo -e "\n[7/7] Testing API server..."
uvicorn src.scripts.deploy_api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API server to start
echo "Waiting for API server to start..."
sleep 10

# Make a test prediction
echo "Making a test prediction..."
curl -X POST "http://localhost:8000/api/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "ph": 7.5,
        "Hardness": 204.89,
        "Solids": 20791.32,
        "Chloramines": 7.3,
        "Sulfate": 368.51,
        "Conductivity": 564.31,
        "Organic_carbon": 10.38,
        "Trihalomethanes": 86.99,
        "Turbidity": 2.96
    }'

echo -e "\nCheck API at http://localhost:8000"
echo "Press Ctrl+C to stop the services"

# Wait for user input
read -p "Press Enter to stop services..."

# Kill background processes
kill $API_PID
kill $MONITORING_PID

echo "Pipeline test complete!"
