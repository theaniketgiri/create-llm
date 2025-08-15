# test-gui

A custom Large Language Model built with create-llm.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train Tokenizer**
   ```bash
   python tokenizer/train_tokenizer.py --input data/raw.txt
   ```

3. **Prepare Dataset**
   ```bash
   python data/prepare_dataset.py
   ```

4. **Train Model**
   ```bash
   python training/train.py --config training/config.yaml
   ```

5. **Evaluate Model**
   ```bash
   python eval/run_eval.py
   ```

6. **Launch Web GUI** (Optional)
   ```bash
   cd web-gui
   npm install
   npm start
   ```
   Then open http://localhost:3001 in your browser

## 📁 Project Structure

- `model/` - Transformer architecture (GPT)
- `tokenizer/` - Tokenizer training scripts (BPE)
- `data/` - Dataset preprocessing (wikitext)
- `training/` - Training pipeline and configuration
- `eval/` - Evaluation scripts and metrics
- `checkpoints/` - Saved model checkpoints
- `logs/` - Training logs and metrics
- `web-gui/` - Web-based training and testing interface

## ⚙️ Configuration

Edit `training/config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Dataset settings
- Logging configuration

## 📊 Monitoring

Training progress is logged to `logs/` directory. You can monitor training in two ways:

### Web GUI (Recommended)
Launch the web interface for real-time training monitoring:
```bash
cd web-gui
npm install
npm start
```
Then open http://localhost:3001 in your browser

### TensorBoard
Use TensorBoard for detailed metrics visualization:
```bash
tensorboard --logdir logs/
```

## 🤝 Contributing

This project was created with [create-llm](https://github.com/theaniketgiri/create-llm).

## 📚 Documentation

For detailed documentation, visit: https://github.com/theaniketgiri/create-llm


## 🤖 Synthetic Data

This project includes synthetic data generation capabilities powered by [SynthexAI](https://synthex.theaniketgiri.me).

Use the synthetic data generation script to create custom datasets:
```bash
python scripts/generate_synthetic_data.py --type medical --size 10000
```

Available data types: medical, code, news, fiction, technical

