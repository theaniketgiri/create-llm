#!/usr/bin/env node

/**
 * Demo script to test the create-llm CLI
 * This creates a test project with default settings
 */

const { generateProject } = require('./src/generator');
const fs = require('fs-extra');
const path = require('path');

async function runDemo() {
  console.log('🚀 Create LLM Demo - Testing CLI functionality\n');
  
  const demoProjectName = 'demo-llm-project';
  const demoProjectPath = path.resolve(demoProjectName);
  
  // Clean up any existing demo project
  if (await fs.pathExists(demoProjectPath)) {
    console.log('Cleaning up existing demo project...');
    await fs.remove(demoProjectPath);
  }
  
  // Default options for demo
  const demoOptions = {
    template: 'gpt',
    tokenizer: 'bpe',
    dataset: 'wikitext',
    useTypescript: false,
    includeSyntheticData: true
  };
  
  console.log('Creating demo project with options:', demoOptions);
  
  try {
    // Generate the project
    await generateProject(demoProjectName, demoOptions);
    
    console.log('\n✅ Demo project created successfully!');
    
    // Verify key files exist
    const keyFiles = [
      'README.md',
      'requirements.txt',
      'setup.py',
      '.gitignore',
      'model/__init__.py',
      'tokenizer/train_tokenizer.py',
      'data/dataset.py',
      'training/train.py',
      'eval/run_eval.py',
      'scripts/generate_synthetic_data.py'
    ];
    
    console.log('\n📁 Verifying project structure:');
    for (const file of keyFiles) {
      const filePath = path.join(demoProjectPath, file);
      const exists = await fs.pathExists(filePath);
      console.log(`  ${exists ? '✅' : '❌'} ${file}`);
    }
    
    // Show project info
    const readmePath = path.join(demoProjectPath, 'README.md');
    const readmeContent = await fs.readFile(readmePath, 'utf-8');
    
    console.log('\n📋 Project Summary:');
    console.log(`  Project name: ${demoProjectName}`);
    console.log(`  Architecture: ${demoOptions.template.toUpperCase()}`);
    console.log(`  Tokenizer: ${demoOptions.tokenizer.toUpperCase()}`);
    console.log(`  Dataset: ${demoOptions.dataset}`);
    console.log(`  Synthetic data: ${demoOptions.includeSyntheticData ? 'Yes' : 'No'}`);
    
    console.log('\n🎉 Demo completed successfully!');
    console.log(`\nTo explore the generated project:`);
    console.log(`  cd ${demoProjectName}`);
    console.log(`  ls -la`);
    console.log(`  cat README.md`);
    
    console.log('\nTo clean up the demo project:');
    console.log(`  rm -rf ${demoProjectName}`);
    
  } catch (error) {
    console.error('\n❌ Demo failed:', error.message);
    process.exit(1);
  }
}

// Run the demo if this file is executed directly
if (require.main === module) {
  runDemo();
}

module.exports = { runDemo }; 