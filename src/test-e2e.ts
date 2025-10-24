#!/usr/bin/env node

/**
 * End-to-end test for create-llm
 * Tests complete project generation and validates all components
 */

import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { ScaffolderEngine } from './scaffolder';
import { TemplateManager } from './template-manager';
import { ConfigGenerator } from './config-generator';
import { ProjectConfig } from './prompts';

console.log(chalk.blue.bold('\n🧪 Running End-to-End Tests\n'));

const testProjectName = 'test-e2e-project';
const testProjectPath = path.join(process.cwd(), testProjectName);

let passed = 0;
let failed = 0;

// Clean up if exists
if (fs.existsSync(testProjectPath)) {
  fs.rmSync(testProjectPath, { recursive: true, force: true });
}

async function runTests() {
  try {
    // Test 1: Template Manager
    console.log(chalk.cyan('Test 1: Template Manager'));
    const templateManager = new TemplateManager();
    const templates = templateManager.getAvailableTemplates();
    
    if (templates.length === 4 && templates.includes('tiny') && templates.includes('small')) {
      console.log(chalk.green('✓ Template Manager working'));
      passed++;
    } else {
      console.log(chalk.red('✗ Template Manager failed'));
      failed++;
    }

    // Test 2: Config Generator
    console.log(chalk.cyan('\nTest 2: Config Generator'));
    const configGenerator = new ConfigGenerator();
    const config: ProjectConfig = {
      projectName: testProjectName,
      template: 'tiny',
      tokenizer: 'bpe',
      plugins: ['wandb'],
      skipInstall: true
    };
    const template = templateManager.getTemplate('tiny');
    const configContent = configGenerator.generateConfig(config, template);
    
    if (configContent.includes('module.exports') && 
        configContent.includes('model:') && 
        configContent.includes('training:')) {
      console.log(chalk.green('✓ Config Generator working'));
      passed++;
    } else {
      console.log(chalk.red('✗ Config Generator failed'));
      failed++;
    }

    // Test 3: Project Scaffolding
    console.log(chalk.cyan('\nTest 3: Project Scaffolding'));
    const scaffolder = new ScaffolderEngine(testProjectPath);
    await scaffolder.createProjectStructure(config, template);
    await scaffolder.copyTemplateFiles(config, template);
    
    if (fs.existsSync(testProjectPath)) {
      console.log(chalk.green('✓ Project directory created'));
      passed++;
    } else {
      console.log(chalk.red('✗ Project directory not created'));
      failed++;
    }

    // Test 4: Directory Structure
    console.log(chalk.cyan('\nTest 4: Directory Structure'));
    const requiredDirs = [
      'data/raw',
      'data/processed',
      'models/architectures',
      'tokenizer',
      'training/callbacks',
      'evaluation',
      'checkpoints',
      'logs',
      'plugins'
    ];
    
    let allDirsExist = true;
    for (const dir of requiredDirs) {
      if (!fs.existsSync(path.join(testProjectPath, dir))) {
        console.log(chalk.red(`✗ Missing directory: ${dir}`));
        allDirsExist = false;
      }
    }
    
    if (allDirsExist) {
      console.log(chalk.green(`✓ All ${requiredDirs.length} directories exist`));
      passed++;
    } else {
      failed++;
    }

    // Test 5: Core Files
    console.log(chalk.cyan('\nTest 5: Core Files'));
    const requiredFiles = [
      'llm.config.js',
      'requirements.txt',
      '.gitignore',
      'README.md',
      'data/raw/sample.txt',
      'data/dataset.py',
      'models/config.py',
      'models/architectures/gpt.py',
      'models/architectures/tiny.py',
      'models/architectures/small.py',
      'models/architectures/base.py',
      'tokenizer/train.py',
      'training/trainer.py',
      'training/callbacks/base.py',
      'training/callbacks/checkpoint.py',
      'training/callbacks/logging.py',
      'training/callbacks/checkpoint_manager.py',
      'training/train.py',
      'evaluation/evaluate.py',
      'evaluation/generate.py',
      'chat.py',
      'deploy.py',
      'compare.py'
    ];
    
    let allFilesExist = true;
    for (const file of requiredFiles) {
      if (!fs.existsSync(path.join(testProjectPath, file))) {
        console.log(chalk.red(`✗ Missing file: ${file}`));
        allFilesExist = false;
      }
    }
    
    if (allFilesExist) {
      console.log(chalk.green(`✓ All ${requiredFiles.length} files exist`));
      passed++;
    } else {
      failed++;
    }

    // Test 6: Config File Content
    console.log(chalk.cyan('\nTest 6: Config File Content'));
    const configPath = path.join(testProjectPath, 'llm.config.js');
    const configFileContent = fs.readFileSync(configPath, 'utf-8');
    
    const configChecks = [
      { check: configFileContent.includes('module.exports'), desc: 'Module exports' },
      { check: configFileContent.includes('model:'), desc: 'Model section' },
      { check: configFileContent.includes('training:'), desc: 'Training section' },
      { check: configFileContent.includes('tokenizer:'), desc: 'Tokenizer section' },
      { check: configFileContent.includes('checkpoints:'), desc: 'Checkpoints section' },
      { check: configFileContent.includes('logging:'), desc: 'Logging section' },
      { check: configFileContent.includes("type: 'bpe'"), desc: 'BPE tokenizer' },
      { check: configFileContent.includes('wandb: true'), desc: 'WandB enabled' }
    ];
    
    let allConfigChecks = true;
    for (const { check, desc } of configChecks) {
      if (!check) {
        console.log(chalk.red(`✗ Config missing: ${desc}`));
        allConfigChecks = false;
      }
    }
    
    if (allConfigChecks) {
      console.log(chalk.green(`✓ Config file valid (${configChecks.length} checks)`));
      passed++;
    } else {
      failed++;
    }

    // Test 7: Python File Syntax
    console.log(chalk.cyan('\nTest 7: Python File Syntax'));
    const pythonFiles = [
      'models/architectures/gpt.py',
      'data/dataset.py',
      'training/trainer.py',
      'tokenizer/train.py'
    ];
    
    let allPythonValid = true;
    for (const file of pythonFiles) {
      const content = fs.readFileSync(path.join(testProjectPath, file), 'utf-8');
      if (!content.includes('import ') || !content.includes('def ')) {
        console.log(chalk.red(`✗ Invalid Python file: ${file}`));
        allPythonValid = false;
      }
    }
    
    if (allPythonValid) {
      console.log(chalk.green(`✓ All Python files have valid syntax`));
      passed++;
    } else {
      failed++;
    }

    // Test 8: Model Architecture
    console.log(chalk.cyan('\nTest 8: Model Architecture'));
    const gptPath = path.join(testProjectPath, 'models/architectures/gpt.py');
    const gptContent = fs.readFileSync(gptPath, 'utf-8');
    
    const modelChecks = [
      { check: gptContent.includes('class GPTModel'), desc: 'GPTModel class' },
      { check: gptContent.includes('class GPTConfig'), desc: 'GPTConfig class' },
      { check: gptContent.includes('class MultiHeadAttention'), desc: 'Attention class' },
      { check: gptContent.includes('class TransformerBlock'), desc: 'Transformer block' },
      { check: gptContent.includes('def forward'), desc: 'Forward method' },
      { check: gptContent.includes('def generate'), desc: 'Generate method' }
    ];
    
    let allModelChecks = true;
    for (const { check, desc } of modelChecks) {
      if (!check) {
        console.log(chalk.red(`✗ Model missing: ${desc}`));
        allModelChecks = false;
      }
    }
    
    if (allModelChecks) {
      console.log(chalk.green(`✓ Model architecture complete (${modelChecks.length} checks)`));
      passed++;
    } else {
      failed++;
    }

    // Test 9: Dataset Implementation
    console.log(chalk.cyan('\nTest 9: Dataset Implementation'));
    const datasetPath = path.join(testProjectPath, 'data/dataset.py');
    const datasetContent = fs.readFileSync(datasetPath, 'utf-8');
    
    const datasetChecks = [
      { check: datasetContent.includes('class LLMDataset'), desc: 'LLMDataset class' },
      { check: datasetContent.includes('def __len__'), desc: '__len__ method' },
      { check: datasetContent.includes('def __getitem__'), desc: '__getitem__ method' },
      { check: datasetContent.includes('input_ids'), desc: 'input_ids' },
      { check: datasetContent.includes('attention_mask'), desc: 'attention_mask' },
      { check: datasetContent.includes('labels'), desc: 'labels' }
    ];
    
    let allDatasetChecks = true;
    for (const { check, desc } of datasetChecks) {
      if (!check) {
        console.log(chalk.red(`✗ Dataset missing: ${desc}`));
        allDatasetChecks = false;
      }
    }
    
    if (allDatasetChecks) {
      console.log(chalk.green(`✓ Dataset implementation complete (${datasetChecks.length} checks)`));
      passed++;
    } else {
      failed++;
    }

    // Test 10: Trainer Implementation
    console.log(chalk.cyan('\nTest 10: Trainer Implementation'));
    const trainerPath = path.join(testProjectPath, 'training/trainer.py');
    const trainerContent = fs.readFileSync(trainerPath, 'utf-8');
    
    const trainerChecks = [
      { check: trainerContent.includes('class Trainer'), desc: 'Trainer class' },
      { check: trainerContent.includes('def train'), desc: 'train method' },
      { check: trainerContent.includes('def evaluate'), desc: 'evaluate method' },
      { check: trainerContent.includes('_create_optimizer'), desc: 'optimizer setup' },
      { check: trainerContent.includes('_create_scheduler'), desc: 'scheduler setup' },
      { check: trainerContent.includes('gradient_accumulation'), desc: 'gradient accumulation' },
      { check: trainerContent.includes('mixed_precision'), desc: 'mixed precision' }
    ];
    
    let allTrainerChecks = true;
    for (const { check, desc } of trainerChecks) {
      if (!check) {
        console.log(chalk.red(`✗ Trainer missing: ${desc}`));
        allTrainerChecks = false;
      }
    }
    
    if (allTrainerChecks) {
      console.log(chalk.green(`✓ Trainer implementation complete (${trainerChecks.length} checks)`));
      passed++;
    } else {
      failed++;
    }

    // Summary
    console.log(chalk.yellow('\n\n' + '='.repeat(60)));
    console.log(chalk.yellow(`Test Results: ${passed} passed, ${failed} failed`));
    console.log(chalk.yellow('='.repeat(60)));

    if (failed === 0) {
      console.log(chalk.green.bold('\n✅ All end-to-end tests passed!\n'));
      return 0;
    } else {
      console.log(chalk.red.bold('\n❌ Some tests failed!\n'));
      return 1;
    }

  } catch (error) {
    console.error(chalk.red('\n❌ Test error:'), error instanceof Error ? error.message : String(error));
    if (error instanceof Error && error.stack) {
      console.error(chalk.gray(error.stack));
    }
    return 1;
  } finally {
    // Clean up
    if (fs.existsSync(testProjectPath)) {
      fs.rmSync(testProjectPath, { recursive: true, force: true });
      console.log(chalk.gray('Cleaned up test project'));
    }
  }
}

runTests().then(exitCode => process.exit(exitCode));
