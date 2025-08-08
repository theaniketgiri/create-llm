const { generateProject } = require('../src/generator');
const fs = require('fs-extra');
const path = require('path');

describe('Create LLM CLI', () => {
  const testProjectName = 'test-llm-project';
  const testProjectPath = path.join(__dirname, '..', testProjectName);
  
  afterEach(async () => {
    // Clean up test project
    if (await fs.pathExists(testProjectPath)) {
      await fs.remove(testProjectPath);
    }
  });
  
  test('should create a complete project structure', async () => {
    const options = {
      template: 'gpt',
      tokenizer: 'bpe',
      dataset: 'wikitext',
      useTypescript: false,
      includeSyntheticData: true
    };
    
    await generateProject(testProjectName, options);
    
    // Check that project directory exists
    expect(await fs.pathExists(testProjectPath)).toBe(true);
    
    // Check that all required directories exist
    const requiredDirs = [
      'model',
      'tokenizer',
      'data',
      'training',
      'eval',
      'checkpoints',
      'logs',
      'scripts'
    ];
    
    for (const dir of requiredDirs) {
      const dirPath = path.join(testProjectPath, dir);
      expect(await fs.pathExists(dirPath)).toBe(true);
    }
    
    // Check that key files exist
    const requiredFiles = [
      'README.md',
      'requirements.txt',
      'setup.py',
      '.gitignore',
      'model/__init__.py',
      'model/config.py',
      'model/transformer.py',
      'tokenizer/__init__.py',
      'tokenizer/train_tokenizer.py',
      'data/__init__.py',
      'data/dataset.py',
      'training/__init__.py',
      'training/train.py',
      'training/config.yaml',
      'eval/__init__.py',
      'eval/run_eval.py',
      'scripts/generate_synthetic_data.py'
    ];
    
    for (const file of requiredFiles) {
      const filePath = path.join(testProjectPath, file);
      expect(await fs.pathExists(filePath)).toBe(true);
    }
  });
  
  test('should create different model architectures', async () => {
    const architectures = ['gpt', 'mistral', 'rwkv', 'mixtral'];
    
    for (const arch of architectures) {
      const projectName = `test-${arch}-project`;
      const projectPath = path.join(__dirname, '..', projectName);
      
      const options = {
        template: arch,
        tokenizer: 'bpe',
        dataset: 'wikitext',
        useTypescript: false,
        includeSyntheticData: false
      };
      
      await generateProject(projectName, options);
      
      // Check that model config contains correct architecture
      const configPath = path.join(projectPath, 'model', 'config.py');
      const configContent = await fs.readFile(configPath, 'utf-8');
      expect(configContent).toContain(`model_type: str = "${arch}"`);
      
      // Clean up
      await fs.remove(projectPath);
    }
  });
  
  test('should create different tokenizer types', async () => {
    const tokenizers = ['bpe', 'wordpiece', 'unigram'];
    
    for (const tokenizer of tokenizers) {
      const projectName = `test-${tokenizer}-tokenizer`;
      const projectPath = path.join(__dirname, '..', projectName);
      
      const options = {
        template: 'gpt',
        tokenizer: tokenizer,
        dataset: 'wikitext',
        useTypescript: false,
        includeSyntheticData: false
      };
      
      await generateProject(projectName, options);
      
      // Check that tokenizer script contains correct type
      const tokenizerPath = path.join(projectPath, 'tokenizer', 'train_tokenizer.py');
      const tokenizerContent = await fs.readFile(tokenizerPath, 'utf-8');
      expect(tokenizerContent).toContain(`default="${tokenizer}"`);
      
      // Clean up
      await fs.remove(projectPath);
    }
  });
}); 