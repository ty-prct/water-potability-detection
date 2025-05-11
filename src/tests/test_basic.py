#!/usr/bin/env python

import os
import pytest


def test_project_structure():
    """Test that essential project directories exist"""
    assert os.path.isdir("src"), "src directory not found"
    assert os.path.isdir("src/scripts"), "src/scripts directory not found"
    assert os.path.isdir("src/tests"), "src/tests directory not found"


def test_script_files_exist():
    """Test that essential script files exist"""
    script_files = [
        "src/scripts/train_pipeline.py",
        "src/scripts/evaluate_pipeline.py",
        "src/scripts/deploy_api.py",
        "src/scripts/model_monitoring.py",
    ]
    
    for script_file in script_files:
        assert os.path.isfile(script_file), f"{script_file} not found"


def test_config_files_exist():
    """Test that configuration files exist"""
    config_files = [
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "Dockerfile",
        "docker-compose.yml",
    ]
    
    for config_file in config_files:
        assert os.path.isfile(config_file), f"{config_file} not found"


def test_directory_permissions():
    """Test that directories have proper permissions to be accessed"""
    directories = [
        "src",
        "src/scripts",
        "src/tests",
        "models",
        "results",
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            assert os.access(directory, os.R_OK), f"Directory {directory} is not readable"


def test_basic_imports():
    """Test importing basic packages needed for the project"""
    try:
        import pandas
        import numpy
        import sklearn
        import fastapi
    except ImportError as e:
        pytest.fail(f"Failed to import required package: {e}")
    
    # This should always pass
    assert True, "Basic imports test"
