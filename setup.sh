#!/bin/bash
# Setup script for water potability MLOps project

# Set up virtual environment
echo "Setting up Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p monitoring/prometheus monitoring/grafana/provisioning logs

# Download sample data if not exists
if [ ! -f "data/train_data.csv" ]; then
  echo "Sample data not found. Downloading..."
  # This would typically download from a data source
  # For the purpose of this script, we assume data exists
  echo "NOTE: Please ensure your data files are available in the data/ directory"
fi

# Set up pre-commit hooks
if [ -d ".git" ]; then
  echo "Setting up pre-commit hooks..."
  pip install pre-commit
  pre-commit install
fi

# Create config symlink for best model
echo "Linking best model..."
if [ -f "results/best_model.pkl" ]; then
  echo "Best model already exists"
else
  # Find the most recent best model
  BEST_MODEL=$(ls -1t results/best_model_* 2>/dev/null | head -n 1)
  if [ -n "$BEST_MODEL" ]; then
    ln -sf "$BEST_MODEL" results/best_model.pkl
    echo "Created symlink to latest best model: $BEST_MODEL"
  else
    echo "No best model found. You may need to train one first."
  fi
fi

# Check Docker availability
if command -v docker &> /dev/null; then
  echo "Docker is installed. You can build the container with: docker-compose build"
else
  echo "Docker not found. For containerized deployment, please install Docker and Docker Compose."
fi

echo "Setup complete! You can now:"
echo "1. Run training pipeline: python src/scripts/train_pipeline.py"
echo "2. Evaluate models: python src/scripts/evaluate_pipeline.py"
echo "3. Start API server: uvicorn src.scripts.deploy_api:app --reload"
echo "4. Start monitoring: python src/scripts/model_monitoring.py"
echo "5. Build and start with Docker: docker-compose up --build"
