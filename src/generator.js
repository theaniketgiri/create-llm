const fs = require('fs-extra');
const path = require('path');
const { createModelFiles } = require('./templates/model');
const { createTokenizerFiles } = require('./templates/tokenizer');
const { createDataFiles } = require('./templates/data');
const { createTrainingFiles } = require('./templates/training');
const { createEvalFiles } = require('./templates/eval');
const { createConfigFiles } = require('./templates/config');

async function generateProject(projectName, options) {
  const projectPath = path.resolve(projectName);
  
  // Create project directory
  await fs.ensureDir(projectPath);
  
  // Create all subdirectories
  const dirs = [
    'model',
    'tokenizer', 
    'data',
    'training',
    'eval',
    'checkpoints',
    'logs',
    'scripts'
  ];
  
  for (const dir of dirs) {
    await fs.ensureDir(path.join(projectPath, dir));
  }
  
  // Generate all project files
  await createModelFiles(projectPath, options);
  await createTokenizerFiles(projectPath, options);
  await createDataFiles(projectPath, options);
  await createTrainingFiles(projectPath, options);
  await createEvalFiles(projectPath, options);
  await createConfigFiles(projectPath, options);
  
  // Create README
  await createREADME(projectPath, projectName, options);
}

async function createREADME(projectPath, projectName, options) {
  const readmeContent = `# ${projectName}

A custom Large Language Model built with create-llm.

## 🚀 Quick Start

1. **Setup Environment**
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   \`\`\`

2. **Train Tokenizer**
   \`\`\`bash
   python tokenizer/train_tokenizer.py --input data/raw.txt
   \`\`\`

3. **Prepare Dataset**
   \`\`\`bash
   python data/prepare_dataset.py
   \`\`\`

4. **Train Model**
   \`\`\`bash
   python training/train.py --config training/config.yaml
   \`\`\`

5. **Evaluate Model**
   \`\`\`bash
   python eval/run_eval.py
   \`\`\`

## 📁 Project Structure

- \`model/\` - Transformer architecture (${options.template.toUpperCase()})
- \`tokenizer/\` - Tokenizer training scripts (${options.tokenizer.toUpperCase()})
- \`data/\` - Dataset preprocessing (${options.dataset})
- \`training/\` - Training pipeline and configuration
- \`eval/\` - Evaluation scripts and metrics
- \`checkpoints/\` - Saved model checkpoints
- \`logs/\` - Training logs and metrics

## ⚙️ Configuration

Edit \`training/config.yaml\` to customize:
- Model architecture parameters
- Training hyperparameters
- Dataset settings
- Logging configuration

## 📊 Monitoring

Training progress is logged to \`logs/\` directory. Use TensorBoard to visualize:

\`\`\`bash
tensorboard --logdir logs/
\`\`\`

## 🤝 Contributing

This project was created with [create-llm](https://github.com/theaniketgiri/create-llm).

## 📚 Documentation

For detailed documentation, visit: https://github.com/theaniketgiri/create-llm
`;

  await fs.writeFile(path.join(projectPath, 'README.md'), readmeContent);
}

module.exports = { generateProject }; 