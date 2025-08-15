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
    'scripts',
    'web-gui'
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
  
  // Create Web GUI
  await createWebGUI(projectPath, options);
  
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

6. **Launch Web GUI** (Optional)
   \`\`\`bash
   cd web-gui
   npm install
   npm start
   \`\`\`
   Then open http://localhost:3001 in your browser

## 📁 Project Structure

- \`model/\` - Transformer architecture (${options.template.toUpperCase()})
- \`tokenizer/\` - Tokenizer training scripts (${options.tokenizer.toUpperCase()})
- \`data/\` - Dataset preprocessing (${options.dataset})
- \`training/\` - Training pipeline and configuration
- \`eval/\` - Evaluation scripts and metrics
- \`checkpoints/\` - Saved model checkpoints
- \`logs/\` - Training logs and metrics
- \`web-gui/\` - Web-based training and testing interface

## ⚙️ Configuration

Edit \`training/config.yaml\` to customize:
- Model architecture parameters
- Training hyperparameters
- Dataset settings
- Logging configuration

## 📊 Monitoring

Training progress is logged to \`logs/\` directory. You can monitor training in two ways:

### Web GUI (Recommended)
Launch the web interface for real-time training monitoring:
\`\`\`bash
cd web-gui
npm install
npm start
\`\`\`
Then open http://localhost:3001 in your browser

### TensorBoard
Use TensorBoard for detailed metrics visualization:
\`\`\`bash
tensorboard --logdir logs/
\`\`\`

## 🤝 Contributing

This project was created with [create-llm](https://github.com/theaniketgiri/create-llm).

## 📚 Documentation

For detailed documentation, visit: https://github.com/theaniketgiri/create-llm
${options.includeSyntheticData ? `

## 🤖 Synthetic Data

This project includes synthetic data generation capabilities powered by [SynthexAI](https://synthex.theaniketgiri.me).

Use the synthetic data generation script to create custom datasets:
\`\`\`bash
python scripts/generate_synthetic_data.py --type medical --size 10000
\`\`\`

Available data types: medical, code, news, fiction, technical
` : ''}
`;

  await fs.writeFile(path.join(projectPath, 'README.md'), readmeContent);
}

async function createWebGUI(projectPath, options) {
  // Copy web GUI files
  const guiSourcePath = path.join(__dirname, '..', 'web-gui');
  const guiDestPath = path.join(projectPath, 'web-gui');
  
  try {
    console.log(`📁 Copying GUI from: ${guiSourcePath}`);
    console.log(`📁 Copying GUI to: ${guiDestPath}`);
    
    // Check if source exists
    const sourceExists = await fs.pathExists(guiSourcePath);
    if (!sourceExists) {
      console.log('❌ Web GUI source directory does not exist');
      return;
    }
    
    await fs.copy(guiSourcePath, guiDestPath);
    console.log('✅ Web GUI files copied successfully');
  } catch (error) {
    console.log('⚠️  Could not copy web GUI files:', error.message);
    console.log('Error details:', error);
  }
}

module.exports = { generateProject }; 