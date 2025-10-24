# 🚀 Quick Start: Interactive Tools

## What You Have

Your `create-llm` scaffolder generates **7 interactive tools**:

```
test-fixed-llm/
├── 📊 training/train.py --dashboard    # Live web dashboard
├── 💬 chat.py                          # Interactive chat
├── 🔍 compare.py                       # Model comparison
├── 🚀 deploy.py                        # Deployment tool
├── 📈 evaluation/evaluate.py           # Evaluation dashboard
├── ✨ evaluation/generate.py           # Text generation
└── 🧪 pytest                           # Testing dashboard
```

---

## 🎯 Most Popular: Live Training Dashboard

### Start it:
```bash
cd test-fixed-llm
python training/train.py --dashboard
```

### Open browser:
```
http://localhost:5000
```

### What you'll see:

```
┌─────────────────────────────────────────────────────────┐
│  🚀 LLM Training Dashboard                              │
│  Status: ● Training  |  Step: 1234/10000               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  📊 Training Metrics                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Loss: 2.456  ↓                                 │   │
│  │  Learning Rate: 0.0003                          │   │
│  │  Tokens/sec: 1,234                              │   │
│  │  GPU Memory: 2.1GB / 4.3GB                      │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  📈 Loss Chart                                          │
│  │                                                      │
│  │  3.0 ┤                                              │
│  │      │╲                                             │
│  │  2.5 ┤ ╲                                            │
│  │      │  ╲___                                        │
│  │  2.0 ┤      ╲___                                    │
│  │      │          ╲___                                │
│  │  1.5 ┤              ╲___                            │
│  │      └──────────────────────────────────────        │
│  │      0    2000   4000   6000   8000  10000          │
│  │                    Steps                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  🎯 Recent Checkpoints                                  │
│  ✓ checkpoint-2000.pt  (loss: 2.456)                   │
│  ✓ checkpoint-4000.pt  (loss: 2.123)                   │
│  ✓ checkpoint-best.pt  (loss: 2.123) ⭐                │
└─────────────────────────────────────────────────────────┘
```

**Features:**
- ✅ Real-time updates (every second)
- ✅ Beautiful charts with Chart.js
- ✅ GPU memory monitoring
- ✅ Checkpoint tracking
- ✅ Responsive design (works on mobile!)

---

## 💬 Interactive Chat

### Start it:
```bash
python chat.py --checkpoint checkpoints/final.pt
```

### Example session:
```
============================================================
🤖 LLM Chat Interface
============================================================
Model loaded: 23M parameters on cuda

You: Hello! Tell me about machine learning.

🤖 Generating... ▓▓▓▓▓▓▓▓▓▓ Done!

Bot: Machine learning is a fascinating field of artificial 
intelligence that enables computers to learn from data without 
being explicitly programmed. It involves algorithms that can 
identify patterns and make predictions...

You: Can you explain neural networks?

🤖 Generating... ▓▓▓▓▓▓▓▓▓▓ Done!

Bot: Neural networks are computational models inspired by the 
human brain. They consist of interconnected nodes (neurons) 
organized in layers...

Commands:
  /temp 0.7    - Adjust creativity
  /clear       - New conversation
  /save        - Save chat history
  /quit        - Exit
```

---

## 🔍 Model Comparison

### Compare two models:
```bash
python compare.py checkpoints/model-v1/ checkpoints/model-v2/
```

### Output:
```
============================================================
🔍 Comparing Models
============================================================

┌─────────────────┬──────────────┬──────────────┐
│ Metric          │ Model V1     │ Model V2     │
├─────────────────┼──────────────┼──────────────┤
│ Loss            │ 2.456        │ 2.123 ✓      │
│ Perplexity      │ 11.65        │ 8.35 ✓       │
│ Accuracy        │ 42.3%        │ 48.7% ✓      │
│ Speed (tok/s)   │ 1234 ✓       │ 1189         │
└─────────────────┴──────────────┴──────────────┘

Sample Generations (Prompt: "Once upon a time"):

Model V1:
"Once upon a time, there was a programmer who..."

Model V2:
"Once upon a time in a distant land, a curious developer..."

🏆 Winner: Model V2 (Better perplexity and accuracy)
```

---

## 🚀 Deployment Tool

### Deploy to Hugging Face:
```bash
python deploy.py --to huggingface --repo-id username/my-model
```

### Interactive prompts:
```
============================================================
🚀 Deploying to Hugging Face Hub
============================================================

Repository: username/my-model
Private: No
Model card: Yes

Uploading files...
  ✓ model.pt (88.2 MB)
  ✓ tokenizer.json (1.2 MB)
  ✓ config.json (2.1 KB)
  ✓ README.md (4.5 KB)

🎉 Deployment complete!

View your model at:
https://huggingface.co/username/my-model

Try it:
from transformers import AutoModel
model = AutoModel.from_pretrained("username/my-model")
```

---

## ✨ Text Generation

### Generate text:
```bash
python evaluation/generate.py --prompt "The future of AI is"
```

### Interactive mode:
```bash
python evaluation/generate.py --interactive
```

```
============================================================
✨ Text Generation Tool
============================================================

Settings:
  Temperature: 0.8 (creativity)
  Top-k: 50 (diversity)
  Top-p: 0.95 (nucleus sampling)
  Max length: 100 tokens

Enter prompt: The future of AI is

Generating... ▓▓▓▓▓▓▓▓▓▓ 100%

The future of AI is incredibly promising. As we continue to 
develop more sophisticated algorithms and gather larger datasets, 
we'll see AI systems that can understand context, reason about 
complex problems, and assist humans in ways we can barely imagine 
today. The key will be ensuring these systems are developed 
responsibly and ethically.

Adjust settings:
  /temp 1.2    - More creative
  /temp 0.3    - More focused
  /topk 100    - More diverse
  /length 200  - Longer output
```

---

## 📊 Evaluation Dashboard

### Evaluate your model:
```bash
python evaluation/evaluate.py --checkpoint checkpoints/final.pt
```

### Results:
```
============================================================
📊 Model Evaluation Report
============================================================

Dataset: data/processed/val.pt (512 examples)

Performance Metrics:
  ✓ Perplexity: 11.65
  ✓ Loss: 2.456
  ✓ Accuracy: 45.2%
  ✓ Speed: 832 tokens/second

Quality Analysis:
  ✓ Grammar score: 87%
  ✓ Coherence score: 82%
  ✓ Diversity score: 76%

Top 5 Best Predictions:
  1. "The quick brown fox..." (loss: 0.23)
  2. "Machine learning is..." (loss: 0.45)
  ...

Top 5 Worst Predictions:
  1. "Quantum mechanics of..." (loss: 8.23)
  2. "In the year 2050..." (loss: 7.45)
  ...

Report saved to: evaluation/report_2024-01-24.json
```

---

## 🧪 Testing Dashboard

### Run all tests:
```bash
pytest --verbose --cov
```

### Output:
```
============================================================
🧪 Test Suite
============================================================

tests/test_model.py
  ✓ test_model_initialization
  ✓ test_model_forward_pass
  ✓ test_model_backward_pass
  ✓ test_model_save_load

tests/test_tokenizer.py
  ✓ test_tokenizer_encode
  ✓ test_tokenizer_decode
  ✓ test_special_tokens

tests/test_data.py
  ✓ test_dataset_loading
  ✓ test_dataloader_batching
  ✓ test_data_preprocessing

tests/test_training.py
  ✓ test_trainer_initialization
  ✓ test_training_step
  ✓ test_checkpoint_saving

============================================================
✓ 12 passed in 5.67s
============================================================

Coverage Report:
  models/        92%
  data/          88%
  training/      85%
  evaluation/    79%
  Overall:       87%

HTML report: htmlcov/index.html
```

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Training with dashboard
python training/train.py --dashboard

# Chat with model
python chat.py --checkpoint checkpoints/final.pt

# Compare models
python compare.py model1/ model2/

# Deploy to HuggingFace
python deploy.py --to huggingface --repo-id user/model

# Evaluate model
python evaluation/evaluate.py --checkpoint checkpoints/final.pt

# Generate text
python evaluation/generate.py --prompt "Your prompt"

# Run tests
pytest --verbose --cov

# Resume training
python training/train.py --resume checkpoints/checkpoint-1000.pt

# Train with custom config
python training/train.py --config custom-config.js
```

---

## 💡 Pro Tips

1. **Dashboard on Remote Server**: Use `--host 0.0.0.0` to access from other devices
   ```bash
   python training/train.py --dashboard --host 0.0.0.0
   ```

2. **Save Chat Conversations**: Use `/save` command in chat to export
   ```
   You: /save my-conversation.txt
   ✓ Conversation saved!
   ```

3. **Batch Generation**: Generate multiple samples at once
   ```bash
   python evaluation/generate.py --prompt "Hello" --num-samples 5
   ```

4. **Custom Dashboard Port**: Change port if 5000 is busy
   ```bash
   python training/train.py --dashboard --port 8080
   ```

5. **Compare Multiple Models**: Compare more than 2
   ```bash
   python compare.py model1/ model2/ model3/
   ```

---

## 🎨 Customization

All tools are customizable! Edit:
- `training/dashboard/templates/dashboard.html` - Dashboard UI
- `chat.py` - Chat interface
- `compare.py` - Comparison metrics
- `llm.config.js` - All configurations

---

Enjoy your interactive LLM training experience! 🚀
