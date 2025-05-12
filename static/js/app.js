// Global variables
let featureImportanceChart = null;
let metricsChart = null;

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const resultSection = document.getElementById('resultSection');
const metricsList = document.getElementById('metricsList');
const refreshMetricsBtn = document.getElementById('refreshMetricsBtn');
const noFeatureImportanceEl = document.getElementById('noFeatureImportance');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Load initial metrics
    loadModelMetrics();

    // Form submission
    predictionForm.addEventListener('submit', handleFormSubmit);

    // Refresh metrics button
    refreshMetricsBtn.addEventListener('click', loadModelMetrics);
});

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();

    // Show loading state
    resultSection.innerHTML = `
        <div class="text-center p-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Processing prediction...</p>
        </div>
    `;

    // Gather form data
    const formData = new FormData(predictionForm);
    const jsonData = {};

    formData.forEach((value, key) => {
        jsonData[key] = parseFloat(value);
    });

    try {
        // Call API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jsonData)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        displayPredictionResult(data);
    } catch (error) {
        console.error('Error making prediction:', error);
        resultSection.innerHTML = `
            <div class="alert alert-danger">
                <h5>Error</h5>
                <p>There was an error making the prediction. Please try again.</p>
                <p class="small">${error.message}</p>
            </div>
        `;
    }
}

// Display prediction results
function displayPredictionResult(data) {
    const isPotable = data.prediction === 'Potable';
    const probability = data.probability * 100;

    resultSection.innerHTML = `
        <div class="result-box ${isPotable ? 'result-potable' : 'result-non-potable'}">
            <h4>Water is ${data.prediction}</h4>
            <p>Confidence: ${probability.toFixed(2)}%</p>
        </div>
        <div class="mt-3">
            <p class="mb-1">Probability:</p>
            <div class="probability-gauge">
                <div class="probability-bar" style="width: ${probability}%"></div>
            </div>
        </div>
        <div class="mt-4">
            <p class="mb-0"><strong>Model used:</strong> ${data.model_used}</p>
            <p class="small text-muted">Prediction made at: ${new Date(data.timestamp).toLocaleString()}</p>
        </div>
    `;

    // Display feature importance if available
    if (data.feature_importance && data.feature_importance.length > 0) {
        displayFeatureImportance(data.feature_importance);
    }
}

// Display feature importance chart
function displayFeatureImportance(featureImportance) {
    // Hide the placeholder
    noFeatureImportanceEl.style.display = 'none';

    // Sort by importance
    const sortedFeatures = [...featureImportance].sort((a, b) => b.importance - a.importance);

    // Prepare data for chart
    const features = sortedFeatures.map(item => item.feature);
    const importanceValues = sortedFeatures.map(item => item.importance);

    // Clear previous chart if exists
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }

    // Create new chart
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Feature Importance',
                data: importanceValues,
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Importance'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                }
            }
        }
    });
}

// Load model metrics
async function loadModelMetrics() {
    try {
        const response = await fetch('/api/metrics');

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        displayModelMetrics(data);
    } catch (error) {
        console.error('Error loading metrics:', error);
        metricsList.innerHTML = `
            <tr>
                <td colspan="2" class="text-danger">Error loading metrics: ${error.message}</td>
            </tr>
        `;
    }
}

// Display model metrics
function displayModelMetrics(data) {
    const metrics = data.metrics;
    const model = data.model;

    // Display metrics in table
    metricsList.innerHTML = `
        <tr>
            <td class="fw-bold">Model</td>
            <td>${model}</td>
        </tr>
        <tr>
            <td class="fw-bold">Accuracy</td>
            <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
        </tr>
        <tr>
            <td class="fw-bold">Precision</td>
            <td>${(metrics.precision * 100).toFixed(2)}%</td>
        </tr>
        <tr>
            <td class="fw-bold">Recall</td>
            <td>${(metrics.recall * 100).toFixed(2)}%</td>
        </tr>
        <tr>
            <td class="fw-bold">F1 Score</td>
            <td>${(metrics.f1_score * 100).toFixed(2)}%</td>
        </tr>
        <tr>
            <td class="fw-bold">ROC AUC</td>
            <td>${(metrics.roc_auc * 100).toFixed(2)}%</td>
        </tr>
    `;

    // Display metrics chart
    displayMetricsChart([
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1_score,
        metrics.roc_auc
    ]);
}

// Display metrics chart
function displayMetricsChart(values) {
    const labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'];

    // Clear previous chart if exists
    if (metricsChart) {
        metricsChart.destroy();
    }

    // Create new chart
    const ctx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Model Performance',
                data: values.map(value => value * 100),
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(52, 152, 219, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}
