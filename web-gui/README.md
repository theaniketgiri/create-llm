# Create LLM - Web GUI

A user-friendly web interface for the Create LLM framework that allows you to train and test language models through a simple browser interface.

## Features

- 📁 **File Upload**: Drag and drop training data files
- ⚙️ **Training Configuration**: Adjust model parameters through a visual interface
- 📊 **Real-time Monitoring**: Watch training progress with live logs and metrics
- 🎯 **Text Generation**: Test your trained models with an interactive prompt interface
- 📈 **Model Management**: View and manage your trained model checkpoints

## Quick Start

### 1. Install Dependencies

```bash
cd web-gui
pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
python app.py
```

### 3. Open Your Browser

Navigate to `http://localhost:5000` to access the web interface.

## Usage Guide

### Training a Model

1. **Upload Training Data**:
   - Click on the "Training" tab
   - Select a text file (.txt, .json, or .csv) containing your training data
   - Click "Upload File"

2. **Configure Training Parameters**:
   - Choose your model architecture (GPT-2, Mistral, RWKV)
   - Adjust parameters like vocabulary size, embedding dimensions, layers
   - Set training hyperparameters (batch size, max steps)

3. **Start Training**:
   - Click "Start Training"
   - Monitor progress in real-time through the status panel
   - View training logs and loss metrics

### Generating Text

1. **Switch to Generation Tab**:
   - Click on the "Text Generation" tab

2. **Enter Your Prompt**:
   - Type your prompt in the text area
   - Adjust generation parameters:
     - **Max Length**: Maximum number of tokens to generate
     - **Temperature**: Controls randomness (0.1 = focused, 1.5 = creative)
     - **Top K**: Limits vocabulary to top K most likely tokens
     - **Top P**: Uses nucleus sampling

3. **Generate Text**:
   - Click "Generate Text"
   - View the generated output below

### Managing Models

1. **View Available Models**:
   - Click on the "Models" tab
   - See all trained model checkpoints
   - View file sizes and modification dates

## Configuration

### Model Parameters

- **Model Type**: Choose between GPT-2, Mistral, or RWKV architectures
- **Vocab Size**: Size of the tokenizer vocabulary (default: 50,000)
- **Embedding Dimension**: Size of token embeddings (default: 768)
- **Number of Layers**: Transformer layers (default: 12)
- **Batch Size**: Training batch size (adjust based on GPU memory)
- **Max Steps**: Maximum training iterations

### Generation Parameters

- **Temperature**: Controls output randomness
  - 0.1-0.7: More focused and coherent
  - 0.8-1.2: Balanced creativity
  - 1.3-2.0: More creative and diverse

- **Top K**: Limits token selection to top K candidates
  - Lower values (10-20): More focused
  - Higher values (40-100): More diverse

- **Top P**: Nucleus sampling threshold
  - 0.1-0.5: Conservative sampling
  - 0.6-0.9: Balanced sampling
  - 0.9-1.0: More diverse sampling

## File Structure

```
web-gui/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/
│   └── js/
│       └── app.js        # Frontend JavaScript
└── uploads/              # Uploaded training files
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload training data
- `POST /start_training` - Start model training
- `GET /training_status` - Get training progress
- `POST /generate` - Generate text
- `GET /models` - List available models

## Troubleshooting

### Common Issues

1. **"No trained model found"**:
   - Ensure training has completed successfully
   - Check that model checkpoints exist in `../demo-llm-project/checkpoints/`

2. **Training fails to start**:
   - Verify that training data has been uploaded
   - Check that the demo-llm-project directory exists
   - Ensure all dependencies are installed

3. **Generation produces poor results**:
   - Train for more steps (increase max_steps)
   - Use a larger, higher-quality training dataset
   - Adjust generation parameters (temperature, top_k, top_p)

4. **Web interface not loading**:
   - Check that Flask is running on port 5000
   - Verify no firewall is blocking the connection
   - Try accessing `http://127.0.0.1:5000` instead

### Performance Tips

- **GPU Memory**: Reduce batch size if you encounter CUDA out of memory errors
- **Training Speed**: Use smaller models (fewer layers/dimensions) for faster training
- **Data Quality**: Clean and preprocess your training data for better results
- **Monitoring**: Use the real-time logs to identify training issues early

## Security Notes

- This web interface is intended for local development use
- Do not expose it to the internet without proper authentication
- Be cautious when uploading sensitive training data
- The interface runs with debug mode enabled by default

## Contributing

To contribute to the web GUI:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This web GUI is part of the Create LLM project and follows the same MIT license.