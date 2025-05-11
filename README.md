# Water Potability Detection - MLOps Project

This project implements an end-to-end MLOps pipeline for water potability detection based on water quality parameters. The system uses machine learning to predict whether water is potable or not.

## Features

- **ML Pipeline**: Data preprocessing, model training, and evaluation
- **API Server**: FastAPI-based REST API for model serving
- **Web Interface**: User-friendly web interface for testing the model
- **Monitoring**: Real-time model monitoring for drift detection
- **MLOps**: Continuous integration, deployment, and monitoring
- **Docker**: Containerized deployment for portability

## Project Structure

- `data/`: Training, validation, and test data (DVC tracked)
- `notebooks/`: Jupyter notebooks for data exploration and model development
- `models/`: Trained ML models (DVC tracked)
- `results/`: Evaluation results and best model artifacts
- `src/scripts/`: Python scripts for training, evaluation, and deployment
- `src/tests/`: Unit and integration tests
- `web/`: Frontend web application
- `monitoring/`: Model monitoring configurations and dashboards

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)

### Setup

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/water-potability-detection.git
cd water-potability-detection
```

2. Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

3. Start the application:

```bash
# Option 1: Run with Python
uvicorn src.scripts.deploy_api:app --reload

# Option 2: Run with Docker
docker-compose up --build
```

   If you encounter Docker issues, try the troubleshooting script:
   
   ```bash
   # Make the script executable
   chmod +x docker_troubleshoot.sh
   
   # Run the script
   ./docker_troubleshoot.sh
   ```

   Common Docker issues and solutions:
   - **Connection refused error**: Docker daemon is not running. Start it with `sudo systemctl start docker`
   - **Permission denied**: Add your user to the docker group with `sudo usermod -aG docker $USER` and log out and back in
   - **Missing dependencies**: Install both Docker and Docker Compose

4. Access the web interface at [http://localhost:8000](http://localhost:8000)

## MLOps Workflow

1. **Data Version Control**: Data and models are tracked with DVC
2. **Training Pipeline**: `src/scripts/train_pipeline.py` trains multiple models
3. **Evaluation**: `src/scripts/evaluate_pipeline.py` evaluates and selects the best model
4. **Deployment**: FastAPI server exposes the model via REST API
5. **Monitoring**: Model performance and drift are continuously monitored
6. **CI/CD**: GitHub Actions for testing, training, and deployment

## API Endpoints

- `GET /`: Web interface
- `POST /api/predict`: Predict water potability
- `GET /api/health`: API health check
- `GET /api/metrics`: Model performance metrics
- `GET /metrics`: Prometheus metrics endpoint

## Monitoring

Access the monitoring dashboards:

- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (default login: admin/admin)

## License

This project is licensed under the MIT License - see the LICENSE file for details.