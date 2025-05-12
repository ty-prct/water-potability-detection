# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -q -r requirements.txt

# Copy the application code
COPY . /app/

# Create required directories
RUN mkdir -p /app/models /app/results /app/monitoring /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH="/app/results/best_model.pkl"
ENV LOG_LEVEL="INFO"

# Add a non-root user for security
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check to verify the application is running
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "src.scripts.deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
