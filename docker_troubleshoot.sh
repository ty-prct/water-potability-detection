#!/bin/bash
# Script to troubleshoot Docker issues and prepare for running docker-compose

# Print header
echo "=========================================="
echo "   Docker Troubleshooting Script"
echo "=========================================="
echo

# Check if Docker is installed
echo "Checking if Docker is installed..."
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
else
    echo "✗ Docker is not installed"
    echo "Please install Docker using your package manager:"
    echo "For Ubuntu/Debian: sudo apt-get update && sudo apt-get install docker.io docker-compose"
    echo "For Arch: sudo pacman -S docker docker-compose"
    exit 1
fi

# Check Docker service status
echo
echo "Checking Docker service status..."
if systemctl is-active docker &> /dev/null; then
    echo "✓ Docker service is running"
else
    echo "✗ Docker service is not running"
    echo "Starting Docker service..."
    sudo systemctl start docker
    if [ $? -eq 0 ]; then
        echo "✓ Docker service started successfully"
    else
        echo "✗ Failed to start Docker service"
        echo "Trying to unmask and enable..."
        sudo systemctl unmask docker
        sudo systemctl enable docker
        sudo systemctl start docker
        if [ $? -eq 0 ]; then
            echo "✓ Docker service started successfully after unmask"
        else
            echo "✗ Still failed to start Docker service"
            echo "Please check Docker installation"
            exit 1
        fi
    fi
fi

# Check current user in docker group
echo
echo "Checking if current user is in docker group..."
if groups | grep -q docker; then
    echo "✓ Current user is in the docker group"
else
    echo "✗ Current user is not in the docker group"
    echo "Adding current user to docker group..."
    sudo usermod -aG docker $USER
    echo "✓ User added to docker group"
    echo "⚠ Please log out and log back in for changes to take effect"
    echo "You can try to temporarily apply changes with: newgrp docker"
fi

# Test Docker
echo
echo "Testing Docker with a simple command..."
docker info &> /dev/null
if [ $? -eq 0 ]; then
    echo "✓ Docker is working properly"
else
    echo "✗ Could not connect to Docker daemon"
    echo "Trying to run with sudo..."
    if sudo docker info &> /dev/null; then
        echo "✓ Docker works when run with sudo"
        echo "For now, you may need to run docker commands with sudo until you log out and back in"
        echo "Example: sudo docker-compose up --build"
    else
        echo "✗ Docker is not working even with sudo"
        echo "Please reinstall Docker or restart your system"
        exit 1
    fi
fi

# Check Docker Compose
echo
echo "Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo "✓ docker-compose is installed as standalone"
elif docker compose version &> /dev/null; then
    echo "✓ docker compose is available as a Docker plugin"
else
    echo "✗ Docker Compose is not installed or not working"
    echo "Please install Docker Compose"
    exit 1
fi

# Summary
echo
echo "=========================================="
echo "   Docker Status Summary"
echo "=========================================="
echo "Docker installed: $(if command -v docker &> /dev/null; then echo 'Yes'; else echo 'No'; fi)"
echo "Docker running: $(if systemctl is-active docker &> /dev/null; then echo 'Yes'; else echo 'No'; fi)"
echo "User in docker group: $(if groups | grep -q docker; then echo 'Yes'; else echo 'No'; fi)"
echo "Docker working without sudo: $(if docker info &> /dev/null; then echo 'Yes'; else echo 'No'; fi)"
echo "Docker Compose available: $(if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then echo 'Yes'; else echo 'No'; fi)"
echo

# Instructions
echo "If Docker is working properly, you can run:"
echo "docker-compose up --build"
echo
echo "If you need to use sudo temporarily:"
echo "sudo docker-compose up --build"
echo
echo "Remember to log out and log back in if you were just added to the docker group"
