#!/usr/bin/env node

/**
 * Test script to verify CLI functionality
 */

const { generateProject } = require('./src/generator');
const fs = require('fs-extra');
const path = require('path');

async function testCLI() {
  console.log('🧪 Testing Create LLM CLI...\n');
  
  const testProjectName = 'test-cli-project';
  const testProjectPath = path.resolve(testProjectName);
  
  // Clean up any existing test project
  if (await fs.pathExists(testProjectPath)) {
    console.log('Cleaning up existing test project...');
    await fs.remove(testProjectPath);
  }
  
  // Test with different configurations
  const testConfigs = [
    {
      name: 'GPT with BPE',
      options: {
        template: 'gpt',
        tokenizer: 'bpe',
        dataset: 'wikitext',
        useTypescript: false,
        includeSyntheticData: true
      }
    },
    {
      name: 'Mistral with WordPiece',
      options: {
        template: 'mistral',
        tokenizer: 'wordpiece',
        dataset: 'c4',
        useTypescript: false,
        includeSyntheticData: false
      }
    },
    {
      name: 'RWKV with Unigram',
      options: {
        template: 'rwkv',
        tokenizer: 'unigram',
        dataset: 'custom',
        useTypescript: false,
        includeSyntheticData: true
      }
    }
  ];
  
  for (const config of testConfigs) {
    console.log(`\n📋 Testing: ${config.name}`);
    console.log(`Options:`, config.options);
    
    try {
      const projectName = `${testProjectName}-${config.template}`;
      const projectPath = path.resolve(projectName);
      
      // Clean up
      if (await fs.pathExists(projectPath)) {
        await fs.remove(projectPath);
      }
      
      // Generate project
      await generateProject(projectName, config.options);
      
      // Verify key files exist
      const keyFiles = [
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
        'eval/run_eval.py'
      ];
      
      let allFilesExist = true;
      for (const file of keyFiles) {
        const filePath = path.join(projectPath, file);
        const exists = await fs.pathExists(filePath);
        if (!exists) {
          console.log(`  ❌ Missing: ${file}`);
          allFilesExist = false;
        }
      }
      
      if (allFilesExist) {
        console.log(`  ✅ All files created successfully`);
      } else {
        console.log(`  ❌ Some files are missing`);
      }
      
      // Clean up
      await fs.remove(projectPath);
      
    } catch (error) {
      console.error(`  ❌ Test failed:`, error.message);
    }
  }
  
  console.log('\n🎉 CLI testing completed!');
}

// Run the test
testCLI().catch(console.error); 