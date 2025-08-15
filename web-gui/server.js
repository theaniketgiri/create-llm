const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs-extra');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// File upload configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});
const upload = multer({ storage });

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Get project status
app.get('/api/status', async (req, res) => {
  try {
    const status = {
      hasTokenizer: await fs.pathExists('tokenizer/tokenizer.json'),
      hasModel: await fs.pathExists('checkpoints'),
      hasData: await fs.pathExists('data/processed'),
      config: await fs.readJson('training/config.yaml').catch(() => null)
    };
    res.json(status);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Start training
app.post('/api/train', (req, res) => {
  const { config } = req.body;
  
  // Save updated config
  fs.writeJson('training/config.yaml', config);
  
  // Start training process
  const trainingProcess = spawn('python', ['training/train.py', '--config', 'training/config.yaml']);
  
  trainingProcess.stdout.on('data', (data) => {
    io.emit('training_log', { type: 'stdout', data: data.toString() });
  });
  
  trainingProcess.stderr.on('data', (data) => {
    io.emit('training_log', { type: 'stderr', data: data.toString() });
  });
  
  trainingProcess.on('close', (code) => {
    io.emit('training_complete', { code });
  });
  
  res.json({ message: 'Training started' });
});

// Test model
app.post('/api/test', async (req, res) => {
  const { prompt, modelPath } = req.body;
  
  try {
    // This would call your model inference script
    const testProcess = spawn('python', ['eval/test_model.py', '--prompt', prompt, '--model', modelPath]);
    
    let output = '';
    testProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    testProcess.on('close', (code) => {
      res.json({ response: output.trim() });
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Upload dataset
app.post('/api/upload', upload.single('dataset'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  
  // Move file to data directory
  const targetPath = path.join('data', 'raw', req.file.originalname);
  fs.move(req.file.path, targetPath);
  
  res.json({ message: 'File uploaded successfully', path: targetPath });
});

// Get training logs
app.get('/api/logs', async (req, res) => {
  try {
    const logs = await fs.readFile('logs/training.log', 'utf8');
    res.json({ logs });
  } catch (error) {
    res.json({ logs: 'No logs available' });
  }
});

// Socket connection for real-time updates
io.on('connection', (socket) => {
  console.log('Client connected');
  
  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

server.listen(PORT, () => {
  console.log(`🚀 LLM GUI Server running on http://localhost:${PORT}`);
  console.log(`📊 Open your browser to start training and testing your LLM!`);
});
