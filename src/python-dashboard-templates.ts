/**
 * Python code templates for live training dashboard
 */

export class PythonDashboardTemplates {
  /**
   * Get DashboardServer class
   */
  static getDashboardServer(): string {
    return `"""
Live Training Dashboard Server
Flask-based web server for real-time training visualization
"""

import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import torch


class DashboardServer:
    """
    Live training dashboard server using Flask and SocketIO
    Provides real-time visualization of training metrics
    """
    
    def __init__(self, port: int = 5000, host: str = '127.0.0.1'):
        """
        Initialize dashboard server
        
        Args:
            port: Port to run server on (default: 5000)
            host: Host to bind to (default: 127.0.0.1)
        """
        self.port = port
        self.host = host
        self.app = Flask(__name__, template_folder='training/dashboard/templates')
        self.app.config['SECRET_KEY'] = 'llm-training-dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Metrics storage
        self.metrics_history: List[Dict] = []
        self.current_metrics: Dict = {}
        self.sample_generations: List[str] = []
        self.training_complete = False
        
        # Setup routes and handlers
        self._setup_routes()
        self._setup_socketio_handlers()
        
        # Server thread
        self.server_thread: Optional[threading.Thread] = None
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Serve dashboard HTML"""
            return render_template('dashboard.html')
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return {'status': 'ok', 'training_complete': self.training_complete}
    
    def _setup_socketio_handlers(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            print(f"Dashboard client connected")
            # Send historical data to new client
            emit('history', {
                'metrics': self.metrics_history,
                'samples': self.sample_generations,
                'training_complete': self.training_complete
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"Dashboard client disconnected")
        
        @self.socketio.on('request_update')
        def handle_request_update():
            """Handle client request for current state"""
            emit('current_state', {
                'metrics': self.current_metrics,
                'samples': self.sample_generations[-5:] if self.sample_generations else [],
                'training_complete': self.training_complete
            })
    
    def update_metrics(self, step: int, metrics: Dict):
        """
        Update dashboard with new training metrics
        
        Args:
            step: Current training step
            metrics: Dictionary of metrics (loss, lr, tokens_per_sec, etc.)
        """
        # Add timestamp
        metrics_with_time = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        # Store in history
        self.metrics_history.append(metrics_with_time)
        self.current_metrics = metrics_with_time
        
        # Limit history size to prevent memory issues
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-10000:]
        
        # Emit to connected clients
        self.socketio.emit('metrics_update', metrics_with_time)
    
    def add_sample_generation(self, step: int, prompt: str, generated_text: str):
        """
        Add sample text generation to dashboard
        
        Args:
            step: Current training step
            prompt: Input prompt
            generated_text: Generated text
        """
        sample = {
            'step': step,
            'prompt': prompt,
            'generated_text': generated_text,
            'timestamp': time.time()
        }
        
        self.sample_generations.append(sample)
        
        # Keep only last 50 samples
        if len(self.sample_generations) > 50:
            self.sample_generations = self.sample_generations[-50:]
        
        # Emit to connected clients
        self.socketio.emit('sample_update', sample)
    
    def set_training_complete(self, final_metrics: Optional[Dict] = None):
        """
        Mark training as complete
        
        Args:
            final_metrics: Optional final metrics to display
        """
        self.training_complete = True
        
        completion_data = {
            'complete': True,
            'timestamp': time.time()
        }
        
        if final_metrics:
            completion_data['final_metrics'] = final_metrics
        
        self.socketio.emit('training_complete', completion_data)
    
    def start(self):
        """Start dashboard server in background thread"""
        if self.server_thread and self.server_thread.is_alive():
            print("Dashboard server already running")
            return
        
        def run_server():
            print(f"\\n🌐 Dashboard server starting on http://{self.host}:{self.port}")
            print(f"   Open this URL in your browser to view training progress\\n")
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
                log_output=False
            )
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Give server time to start
        time.sleep(1)
    
    def stop(self):
        """Stop dashboard server"""
        self.training_complete = True
        # SocketIO server will stop when main thread exits (daemon=True)


def create_dashboard_server(port: int = 5000) -> DashboardServer:
    """
    Create and start dashboard server
    
    Args:
        port: Port to run server on
    
    Returns:
        DashboardServer instance
    """
    server = DashboardServer(port=port)
    server.start()
    return server


if __name__ == '__main__':
    # Test dashboard server
    import random
    
    server = create_dashboard_server(port=5000)
    
    print("Testing dashboard with simulated training...")
    
    # Simulate training
    for step in range(100):
        metrics = {
            'loss': 5.0 - (step * 0.04) + random.uniform(-0.1, 0.1),
            'val_loss': 5.2 - (step * 0.03) + random.uniform(-0.1, 0.1),
            'learning_rate': 3e-4 * (1 - step / 100),
            'tokens_per_sec': 1000 + random.uniform(-100, 100),
            'gpu_memory_used': 8.5 + random.uniform(-0.5, 0.5)
        }
        
        server.update_metrics(step, metrics)
        
        # Add sample generation every 10 steps
        if step % 10 == 0:
            server.add_sample_generation(
                step,
                "Once upon a time",
                f"Once upon a time, there was a model at step {step}..."
            )
        
        time.sleep(0.1)
    
    server.set_training_complete({'final_loss': 1.5})
    
    print("\\nTest complete. Dashboard will remain open.")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nShutting down...")
`;
  }

  /**
   * Get DashboardCallback class
   */
  static getDashboardCallback(): string {
    return `"""
Dashboard Callback for Training
Integrates dashboard with training loop
"""

from typing import Optional
from .base import Callback


class DashboardCallback(Callback):
    """
    Callback that updates live training dashboard
    """
    
    def __init__(
        self,
        dashboard_server,
        update_interval: int = 10,
        generate_samples: bool = True,
        sample_prompts: Optional[list] = None,
        sample_interval: int = 500
    ):
        """
        Initialize dashboard callback
        
        Args:
            dashboard_server: DashboardServer instance
            update_interval: Steps between dashboard updates
            generate_samples: Whether to generate sample text
            sample_prompts: List of prompts for sample generation
            sample_interval: Steps between sample generations
        """
        self.dashboard = dashboard_server
        self.update_interval = update_interval
        self.generate_samples = generate_samples
        self.sample_prompts = sample_prompts or [
            "Once upon a time",
            "The future of AI",
            "In a world where",
            "Scientists discovered"
        ]
        self.sample_interval = sample_interval
        self.current_prompt_idx = 0
    
    def on_train_begin(self, trainer):
        """Called when training begins"""
        print("✓ Dashboard callback initialized")
        print(f"  Updates every {self.update_interval} steps")
        if self.generate_samples:
            print(f"  Sample generation every {self.sample_interval} steps")
    
    def on_step_end(self, trainer, step: int, metrics: dict):
        """
        Called after each training step
        
        Args:
            trainer: Trainer instance
            step: Current step
            metrics: Current metrics
        """
        # Update dashboard at specified interval
        if step % self.update_interval == 0:
            # Prepare metrics for dashboard
            dashboard_metrics = {
                'loss': metrics.get('loss', 0.0),
                'learning_rate': metrics.get('lr', 0.0),
                'tokens_per_sec': metrics.get('tokens_per_sec', 0.0),
                'gpu_memory_used': metrics.get('gpu_memory_gb', 0.0),
                'eta_seconds': metrics.get('eta_seconds', 0)
            }
            
            # Add validation loss if available
            if 'val_loss' in metrics:
                dashboard_metrics['val_loss'] = metrics['val_loss']
            
            self.dashboard.update_metrics(step, dashboard_metrics)
        
        # Generate sample text at specified interval
        if self.generate_samples and step % self.sample_interval == 0 and step > 0:
            self._generate_sample(trainer, step)
    
    def on_train_end(self, trainer, final_metrics: dict):
        """
        Called when training ends
        
        Args:
            trainer: Trainer instance
            final_metrics: Final training metrics
        """
        # Send final metrics to dashboard
        self.dashboard.set_training_complete({
            'final_loss': final_metrics.get('loss', 0.0),
            'final_val_loss': final_metrics.get('val_loss', 0.0),
            'total_steps': final_metrics.get('step', 0)
        })
        
        print("✓ Dashboard updated with final metrics")
    
    def _generate_sample(self, trainer, step: int):
        """
        Generate sample text and send to dashboard
        
        Args:
            trainer: Trainer instance
            step: Current step
        """
        try:
            # Get next prompt
            prompt = self.sample_prompts[self.current_prompt_idx]
            self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.sample_prompts)
            
            # Generate text
            trainer.model.eval()
            
            # Simple generation (you may want to use a proper tokenizer)
            # This is a placeholder - actual implementation would use the tokenizer
            generated_text = f"{prompt} [Generated at step {step}]"
            
            trainer.model.train()
            
            # Send to dashboard
            self.dashboard.add_sample_generation(step, prompt, generated_text)
            
        except Exception as e:
            print(f"Warning: Failed to generate sample: {e}")
`;
  }

  /**
   * Get dashboard HTML template
   */
  static getDashboardHTML(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Training Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            background: white;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        h1 {
            color: #667eea;
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .status.training {
            background: #10b981;
            color: white;
        }
        
        .status.complete {
            background: #3b82f6;
            color: white;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            font-size: 18px;
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .metric {
            margin-bottom: 15px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #333;
        }
        
        .metric-value.large {
            font-size: 32px;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 10px;
        }
        
        .samples {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .sample {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        
        .sample-step {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .sample-prompt {
            font-weight: 600;
            margin-bottom: 5px;
            color: #667eea;
        }
        
        .sample-text {
            color: #333;
            line-height: 1.6;
        }
        
        .completion-message {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            display: none;
        }
        
        .completion-message h2 {
            font-size: 32px;
            margin-bottom: 10px;
            color: white;
        }
        
        .completion-message p {
            font-size: 18px;
            opacity: 0.9;
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            background: #ef4444;
            color: white;
            display: none;
        }
        
        .connection-status.connected {
            background: #10b981;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Disconnected</div>
    
    <div class="container">
        <header>
            <h1>🚀 LLM Training Dashboard</h1>
            <span class="status training" id="trainingStatus">Training in Progress</span>
        </header>
        
        <div class="completion-message" id="completionMessage">
            <h2>🎉 Training Complete!</h2>
            <p>Your model has finished training successfully.</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Current Metrics</h2>
                <div class="metric">
                    <div class="metric-label">Training Loss</div>
                    <div class="metric-value large" id="currentLoss">--</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Validation Loss</div>
                    <div class="metric-value" id="currentValLoss">--</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Learning Rate</div>
                    <div class="metric-value" id="currentLR">--</div>
                </div>
            </div>
            
            <div class="card">
                <h2>⚡ Performance</h2>
                <div class="metric">
                    <div class="metric-label">Tokens/Second</div>
                    <div class="metric-value large" id="tokensPerSec">--</div>
                </div>
                <div class="metric">
                    <div class="metric-label">GPU Memory</div>
                    <div class="metric-value" id="gpuMemory">--</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ETA</div>
                    <div class="metric-value" id="eta">--</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📈 Loss Curves</h2>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>✨ Sample Generations</h2>
            <div class="samples" id="samplesContainer">
                <p style="color: #666; text-align: center; padding: 20px;">
                    Waiting for sample generations...
                </p>
            </div>
        </div>
    </div>
    
    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Connection status
        const connectionStatus = document.getElementById('connectionStatus');
        
        socket.on('connect', () => {
            console.log('Connected to dashboard server');
            connectionStatus.textContent = 'Connected';
            connectionStatus.classList.add('connected');
            connectionStatus.style.display = 'block';
            setTimeout(() => {
                connectionStatus.style.display = 'none';
            }, 2000);
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from dashboard server');
            connectionStatus.textContent = 'Disconnected';
            connectionStatus.classList.remove('connected');
            connectionStatus.style.display = 'block';
        });
        
        // Initialize Chart.js
        const ctx = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Loss'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Step'
                        }
                    }
                }
            }
        });
        
        // Handle historical data
        socket.on('history', (data) => {
            console.log('Received historical data');
            
            // Update chart with historical metrics
            if (data.metrics && data.metrics.length > 0) {
                data.metrics.forEach(metric => {
                    updateChart(metric);
                });
            }
            
            // Update samples
            if (data.samples && data.samples.length > 0) {
                data.samples.forEach(sample => {
                    addSample(sample);
                });
            }
            
            // Check if training is complete
            if (data.training_complete) {
                showCompletionMessage();
            }
        });
        
        // Handle metrics updates
        socket.on('metrics_update', (metrics) => {
            updateMetrics(metrics);
            updateChart(metrics);
        });
        
        // Handle sample updates
        socket.on('sample_update', (sample) => {
            addSample(sample);
        });
        
        // Handle training completion
        socket.on('training_complete', (data) => {
            showCompletionMessage(data.final_metrics);
        });
        
        function updateMetrics(metrics) {
            // Update current metrics display
            document.getElementById('currentLoss').textContent = 
                metrics.loss ? metrics.loss.toFixed(4) : '--';
            
            document.getElementById('currentValLoss').textContent = 
                metrics.val_loss ? metrics.val_loss.toFixed(4) : '--';
            
            document.getElementById('currentLR').textContent = 
                metrics.learning_rate ? metrics.learning_rate.toExponential(2) : '--';
            
            document.getElementById('tokensPerSec').textContent = 
                metrics.tokens_per_sec ? Math.round(metrics.tokens_per_sec) : '--';
            
            document.getElementById('gpuMemory').textContent = 
                metrics.gpu_memory_used ? metrics.gpu_memory_used.toFixed(1) + ' GB' : '--';
            
            // Format ETA
            if (metrics.eta_seconds) {
                const hours = Math.floor(metrics.eta_seconds / 3600);
                const minutes = Math.floor((metrics.eta_seconds % 3600) / 60);
                document.getElementById('eta').textContent = 
                    \`\${hours}h \${minutes}m\`;
            } else {
                document.getElementById('eta').textContent = '--';
            }
        }
        
        function updateChart(metrics) {
            const step = metrics.step;
            
            // Add data point
            lossChart.data.labels.push(step);
            lossChart.data.datasets[0].data.push(metrics.loss);
            
            if (metrics.val_loss) {
                lossChart.data.datasets[1].data.push(metrics.val_loss);
            }
            
            // Keep only last 100 points for performance
            if (lossChart.data.labels.length > 100) {
                lossChart.data.labels.shift();
                lossChart.data.datasets[0].data.shift();
                lossChart.data.datasets[1].data.shift();
            }
            
            lossChart.update('none'); // Update without animation for performance
        }
        
        function addSample(sample) {
            const container = document.getElementById('samplesContainer');
            
            // Clear placeholder text
            if (container.querySelector('p')) {
                container.innerHTML = '';
            }
            
            // Create sample element
            const sampleEl = document.createElement('div');
            sampleEl.className = 'sample';
            sampleEl.innerHTML = \`
                <div class="sample-step">Step \${sample.step}</div>
                <div class="sample-prompt">Prompt: \${sample.prompt}</div>
                <div class="sample-text">\${sample.generated_text}</div>
            \`;
            
            // Add to top
            container.insertBefore(sampleEl, container.firstChild);
            
            // Keep only last 10 samples
            while (container.children.length > 10) {
                container.removeChild(container.lastChild);
            }
        }
        
        function showCompletionMessage(finalMetrics) {
            document.getElementById('completionMessage').style.display = 'block';
            document.getElementById('trainingStatus').textContent = 'Training Complete';
            document.getElementById('trainingStatus').classList.remove('training');
            document.getElementById('trainingStatus').classList.add('complete');
            
            if (finalMetrics) {
                const msg = document.getElementById('completionMessage');
                msg.innerHTML += \`
                    <p style="margin-top: 15px;">
                        Final Loss: \${finalMetrics.final_loss ? finalMetrics.final_loss.toFixed(4) : 'N/A'}
                    </p>
                \`;
            }
        }
    </script>
</body>
</html>
`;
  }

  /**
   * Get training dashboard __init__.py
   */
  static getDashboardInit(): string {
    return `"""
Training dashboard package
"""

from .dashboard_server import DashboardServer, create_dashboard_server
from .dashboard_callback import DashboardCallback

__all__ = [
    'DashboardServer',
    'create_dashboard_server',
    'DashboardCallback',
]
`;
  }
}
