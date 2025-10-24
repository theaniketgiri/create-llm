# 🎨 Interactive Tools & GUIs Guide

Your `create-llm` scaffolder generates several interactive tools and GUIs. Here's how to use them all!

## 📊 1. Live Training Dashboard (Web GUI)

A real-time web-based dashboard to monitor training progress.

### How to Use:
```bash
cd test-fixed-llm
python training/train.py --dashboard
```

Then open your browser to: **http://localhost:5000**

### Features:
- 📈 **Real-time loss curves** - Watch training and validation loss
- 🎯 **Learning rate schedule** - See LR changes over time
- ⚡ **Performance metrics** - Tokens/second, GPU memory usage
- 📊 **Training progress** - Steps, epochs, time elapsed
- 🔄 **Auto-refresh** - Updates every second via WebSocket
- 💾 **Checkpoint status** - See when checkpoints are saved

### Dashboard Interface:
```
┌─────────────────────────────────────────────┐
│  🚀 LLM Training Dashboard                  │
├─────────────────────────────────────────────┤
│  Status: Training | Step: 1234/10000       │
│  Loss: 2.456 | LR: 0.0003                  │
│  Tokens/s: 1234 | GPU: 2.1GB/4.0GB         │
├─────────────────────────────────────────────┤
│  [Loss Chart - Real-time Line Graph]       │
│  [LR Chart - Learning Rate Schedule]       │
│  [Performance Chart - Tokens/s]            │
└─────────────────────────────────────────────┘
```

---

## 💬 2. Interactive Chat Interface

Chat with your trained model in the terminal.

### How to Use:
```bash
cd test-fixed-llm
python chat.py --checkpoint checkpoints/final.pt
```

### Features:
- 🗨️ **Interactive conversation** - Type and get responses
- 🎛️ **Adjustable parameters** - Temperature, top-k, top-p
- 📝 **Context maintenance** - Model remembers conversation
- 🔄 **Loading indicators** - Shows when generating
- 💾 **Save conversations** - Export chat history

### Chat Interface:
```
============================================================
🤖 LLM Chat Interface
============================================================
Model: checkpoints/final.pt
Parameters: 23M | Device: cuda

Commands:
  /temp <value>  - Set temperature (0.1-2.0)
  /topk <value>  - Set top-k sampling
  /topp <value>  - Set top-p (nucleus) sampling
  /clear         - Clear conversation history
  /save          - Save conversation
  /quit          - Exit chat

============================================================

You: Hello! How are you?
🤖 Generating...
Bot: I'm doing well, thank you for asking! How can I help you today?

You: _
```

---

## 📊 3. Model Comparison Tool

Compare multiple trained models side-by-side.

### How to Use:
```bash
cd test-fixed-llm
python compare.py model1/ model2/
```

### Features:
- 📈 **Metrics comparison** - Loss, perplexity, accuracy
- 🎯 **Side-by-side generation** - Same prompts, different outputs
- 📊 **Performance stats** - Speed, memory usage
- 📝 **Detailed reports** - Export comparison results
- 🏆 **Winner selection** - Automatic best model detection

### Comparison Output:
```
============================================================
🔍 Model Comparison
============================================================

Model 1: checkpoints/model-v1/
  Parameters: 23M
  Loss: 2.456
  Perplexity: 11.65
  Speed: 1234 tokens/s

Model 2: checkpoints/model-v2/
  Parameters: 23M
  Loss: 2.123
  Perplexity: 8.35
  Speed: 1189 tokens/s

============================================================
Sample Generations
============================================================

Prompt: "Once upon a time"

Model 1: Once upon a time, there was a curious programmer...
Model 2: Once upon a time in a distant land, a developer...

============================================================
Winner: Model 2 (Lower perplexity)
============================================================
```

---

## 🚀 4. Deployment Tool

Interactive deployment to various platforms.

### How to Use:
```bash
cd test-fixed-llm
python deploy.py --to huggingface --repo-id username/my-model
```

### Supported Platforms:
- 🤗 **Hugging Face Hub** - Share models publicly or privately
- 🔄 **Replicate** - Deploy as API endpoint
- 🐳 **Docker** - Containerize your model
- ☁️ **AWS/GCP/Azure** - Cloud deployment

### Interactive Prompts:
```
============================================================
🚀 Model Deployment Tool
============================================================

Select deployment platform:
  1. Hugging Face Hub
  2. Replicate
  3. Docker Container
  4. Cloud (AWS/GCP/Azure)

Choice: 1

Enter repository name: username/my-awesome-llm
Make repository private? (y/n): n
Include model card? (y/n): y

Deploying to Hugging Face Hub...
✓ Model uploaded
✓ Tokenizer uploaded
✓ Config uploaded
✓ Model card generated

🎉 Deployment complete!
View at: https://huggingface.co/username/my-awesome-llm
```

---

## 📈 5. Evaluation Dashboard

Comprehensive model evaluation with visualizations.

### How to Use:
```bash
cd test-fixed-llm
python evaluation/evaluate.py --checkpoint checkpoints/final.pt
```

### Features:
- 📊 **Perplexity calculation** - Model quality metric
- 🎯 **Accuracy metrics** - Token-level accuracy
- 📈 **Loss distribution** - Histogram of losses
- 🔍 **Error analysis** - Common failure patterns
- 📝 **Detailed reports** - Export evaluation results

### Evaluation Output:
```
============================================================
📊 Model Evaluation
============================================================

Checkpoint: checkpoints/final.pt
Dataset: data/processed/val.pt

Metrics:
  Perplexity: 11.65
  Loss: 2.456
  Accuracy: 45.2%
  Tokens evaluated: 10,240

Performance:
  Evaluation time: 12.3s
  Tokens/second: 832

============================================================
Loss Distribution
============================================================
[Histogram showing loss distribution]

Top 10 Most Difficult Examples:
1. Loss: 8.23 - "The quantum mechanics of..."
2. Loss: 7.45 - "In the year 2050..."
...

✓ Evaluation complete!
Report saved to: evaluation/report.json
```

---

## 🎮 6. Text Generation Tool

Interactive text generation with various sampling strategies.

### How to Use:
```bash
cd test-fixed-llm
python evaluation/generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
```

### Features:
- 🎲 **Multiple sampling methods** - Greedy, temperature, top-k, nucleus
- 🎛️ **Interactive controls** - Adjust parameters on the fly
- 📝 **Batch generation** - Generate multiple samples
- 💾 **Save outputs** - Export generated text
- 🔄 **Streaming mode** - See text as it's generated

### Interactive Mode:
```bash
python evaluation/generate.py --interactive
```

```
============================================================
✨ Text Generation Tool
============================================================

Model: checkpoints/final.pt
Parameters: 23M | Device: cuda

Settings:
  Temperature: 0.8
  Top-k: 50
  Top-p: 0.95
  Max length: 100

Commands:
  /temp <value>  - Set temperature
  /topk <value>  - Set top-k
  /topp <value>  - Set top-p
  /length <val>  - Set max length
  /quit          - Exit

============================================================

Enter prompt: Once upon a time

Generating... ▓▓▓▓▓▓▓▓▓▓ 100%

Once upon a time, in a land far away, there lived a curious 
programmer who wanted to train their own language model. They 
gathered text data from various sources and prepared it carefully.

Generate another? (y/n): _
```

---

## 🧪 7. Testing Dashboard

Run comprehensive tests with visual feedback.

### How to Use:
```bash
cd test-fixed-llm
pytest --verbose
```

### Features:
- ✅ **Test coverage** - See what's tested
- 📊 **Progress bars** - Visual test progress
- 🐛 **Failure details** - Detailed error messages
- 📈 **Performance tests** - Speed benchmarks
- 📝 **Test reports** - HTML reports with coverage

### Test Output:
```
============================================================
🧪 Running Tests
============================================================

tests/test_model.py::test_model_forward ✓
tests/test_model.py::test_model_backward ✓
tests/test_tokenizer.py::test_tokenizer_encode ✓
tests/test_tokenizer.py::test_tokenizer_decode ✓
tests/test_data.py::test_dataset_loading ✓
tests/test_training.py::test_trainer_init ✓

============================================================
✓ 6 passed in 2.34s
============================================================

Coverage: 87%
Report: htmlcov/index.html
```

---

## 🎯 Quick Reference

### Start Training with Dashboard:
```bash
python training/train.py --dashboard
# Open http://localhost:5000
```

### Chat with Model:
```bash
python chat.py --checkpoint checkpoints/final.pt
```

### Compare Models:
```bash
python compare.py model1/ model2/
```

### Deploy Model:
```bash
python deploy.py --to huggingface --repo-id username/model
```

### Evaluate Model:
```bash
python evaluation/evaluate.py --checkpoint checkpoints/final.pt
```

### Generate Text:
```bash
python evaluation/generate.py --prompt "Your prompt here"
```

### Run Tests:
```bash
pytest --verbose --cov
```

---

## 🎨 Customization

All these tools can be customized by editing:
- `llm.config.js` - Training and model configuration
- `training/dashboard/templates/dashboard.html` - Dashboard UI
- `chat.py` - Chat interface settings
- `compare.py` - Comparison metrics
- `deploy.py` - Deployment options

---

## 📚 Additional Resources

- **README.md** - Complete project documentation
- **llm.config.js** - Configuration guide with comments
- **plugins/README.md** - Plugin system documentation
- **tests/** - Example test cases

---

## 💡 Tips

1. **Dashboard Performance**: The dashboard updates in real-time but can be resource-intensive. Use `--dashboard-interval 5` to update every 5 seconds instead.

2. **Chat Context**: The chat interface maintains context. Use `/clear` to start fresh conversations.

3. **Model Comparison**: Compare models trained with different hyperparameters to find the best configuration.

4. **Deployment**: Test locally before deploying to production platforms.

5. **Generation Quality**: Adjust temperature (0.7-1.0 for creative, 0.1-0.3 for factual) based on your use case.

---

Enjoy your interactive LLM training experience! 🚀
