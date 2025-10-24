#!/usr/bin/env node

/**
 * Test script for ConfigGenerator
 */

import chalk from 'chalk';
import { ConfigGenerator } from './config-generator';
import { TemplateManager } from './template-manager';
import { ProjectConfig } from './prompts';

console.log(chalk.blue.bold('\n🧪 Testing ConfigGenerator\n'));

const templateManager = new TemplateManager();
const configGenerator = new ConfigGenerator();

let passed = 0;
let failed = 0;

// Test 1: Generate config for tiny template
console.log(chalk.cyan('Test 1: Generate config for tiny template...'));
try {
  const config: ProjectConfig = {
    projectName: 'test-project',
    template: 'tiny',
    tokenizer: 'bpe',
    plugins: [],
    skipInstall: false
  };
  const template = templateManager.getTemplate('tiny');
  const configContent = configGenerator.generateConfig(config, template);

  // Verify content
  const checks = [
    { check: configContent.includes('module.exports'), desc: 'Module exports' },
    { check: configContent.includes("type: 'gpt'"), desc: 'Model type' },
    { check: configContent.includes("size: 'tiny'"), desc: 'Template size' },
    { check: configContent.includes('vocab_size: 32000'), desc: 'Vocab size' },
    { check: configContent.includes('layers: 6'), desc: 'Layers' },
    { check: configContent.includes('heads: 6'), desc: 'Heads' },
    { check: configContent.includes('dim: 384'), desc: 'Dimension' },
    { check: configContent.includes('batch_size: 16'), desc: 'Batch size' },
    { check: configContent.includes('learning_rate: 0.0006'), desc: 'Learning rate' },
    { check: configContent.includes("tokenizer: {"), desc: 'Tokenizer section' },
    { check: configContent.includes("type: 'bpe'"), desc: 'Tokenizer type' },
    { check: configContent.includes('checkpoints: {'), desc: 'Checkpoints section' },
    { check: configContent.includes('logging: {'), desc: 'Logging section' },
    { check: configContent.includes('plugins: ['), desc: 'Plugins section' },
    { check: configContent.includes('deployment: {'), desc: 'Deployment section' }
  ];

  let allChecks = true;
  for (const { check, desc } of checks) {
    if (!check) {
      console.log(chalk.red(`  ✗ Missing: ${desc}`));
      allChecks = false;
      failed++;
    }
  }

  if (allChecks) {
    console.log(chalk.green(`✓ Tiny template config generated correctly (${checks.length} checks)`));
    passed++;
  }
} catch (error) {
  console.log(chalk.red('✗ Failed to generate tiny config'));
  console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
  failed++;
}

// Test 2: Generate config for small template
console.log(chalk.cyan('\nTest 2: Generate config for small template...'));
try {
  const config: ProjectConfig = {
    projectName: 'test-project',
    template: 'small',
    tokenizer: 'wordpiece',
    plugins: [],
    skipInstall: false
  };
  const template = templateManager.getTemplate('small');
  const configContent = configGenerator.generateConfig(config, template);

  const checks = [
    { check: configContent.includes("size: 'small'"), desc: 'Small template' },
    { check: configContent.includes('layers: 12'), desc: 'Small layers' },
    { check: configContent.includes('heads: 12'), desc: 'Small heads' },
    { check: configContent.includes('dim: 768'), desc: 'Small dimension' },
    { check: configContent.includes('batch_size: 32'), desc: 'Small batch size' },
    { check: configContent.includes("type: 'wordpiece'"), desc: 'WordPiece tokenizer' },
    { check: configContent.includes('mixed_precision: true'), desc: 'Mixed precision enabled' }
  ];

  let allChecks = true;
  for (const { check, desc } of checks) {
    if (!check) {
      console.log(chalk.red(`  ✗ Missing: ${desc}`));
      allChecks = false;
      failed++;
    }
  }

  if (allChecks) {
    console.log(chalk.green(`✓ Small template config generated correctly (${checks.length} checks)`));
    passed++;
  }
} catch (error) {
  console.log(chalk.red('✗ Failed to generate small config'));
  console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
  failed++;
}

// Test 3: Generate config with plugins
console.log(chalk.cyan('\nTest 3: Generate config with plugins...'));
try {
  const config: ProjectConfig = {
    projectName: 'test-project',
    template: 'tiny',
    tokenizer: 'bpe',
    plugins: ['wandb', 'huggingface'],
    skipInstall: false
  };
  const template = templateManager.getTemplate('tiny');
  const configContent = configGenerator.generateConfig(config, template);

  const checks = [
    { check: configContent.includes("'wandb'"), desc: 'WandB plugin' },
    { check: configContent.includes("'huggingface'"), desc: 'HuggingFace plugin' },
    { check: configContent.includes('wandb: true'), desc: 'WandB logging enabled' }
  ];

  let allChecks = true;
  for (const { check, desc } of checks) {
    if (!check) {
      console.log(chalk.red(`  ✗ Missing: ${desc}`));
      allChecks = false;
      failed++;
    }
  }

  if (allChecks) {
    console.log(chalk.green(`✓ Config with plugins generated correctly (${checks.length} checks)`));
    passed++;
  }
} catch (error) {
  console.log(chalk.red('✗ Failed to generate config with plugins'));
  console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
  failed++;
}

// Test 4: Generate config with tips
console.log(chalk.cyan('\nTest 4: Generate config with tips...'));
try {
  const config: ProjectConfig = {
    projectName: 'test-project',
    template: 'base',
    tokenizer: 'unigram',
    plugins: [],
    skipInstall: false
  };
  const template = templateManager.getTemplate('base');
  const configContent = configGenerator.generateConfigWithTips(config, template);

  const checks = [
    { check: configContent.includes('Configuration Tips:'), desc: 'Tips section' },
    { check: configContent.includes('Template: BASE'), desc: 'Template name in tips' },
    { check: configContent.includes('Parameters: 1000M'), desc: 'Parameters in tips' },
    { check: configContent.includes('Training Tips:'), desc: 'Training tips section' },
    { check: configContent.includes('Common Adjustments:'), desc: 'Common adjustments section' }
  ];

  let allChecks = true;
  for (const { check, desc } of checks) {
    if (!check) {
      console.log(chalk.red(`  ✗ Missing: ${desc}`));
      allChecks = false;
      failed++;
    }
  }

  if (allChecks) {
    console.log(chalk.green(`✓ Config with tips generated correctly (${checks.length} checks)`));
    passed++;
  }
} catch (error) {
  console.log(chalk.red('✗ Failed to generate config with tips'));
  console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
  failed++;
}

// Test 5: Verify all tokenizer types
console.log(chalk.cyan('\nTest 5: Verify all tokenizer types...'));
try {
  const tokenizers: Array<'bpe' | 'wordpiece' | 'unigram'> = ['bpe', 'wordpiece', 'unigram'];
  let allTokenizersWork = true;

  for (const tokenizerType of tokenizers) {
    const config: ProjectConfig = {
      projectName: 'test-project',
      template: 'tiny',
      tokenizer: tokenizerType,
      plugins: [],
      skipInstall: false
    };
    const template = templateManager.getTemplate('tiny');
    const configContent = configGenerator.generateConfig(config, template);

    if (!configContent.includes(`type: '${tokenizerType}'`)) {
      console.log(chalk.red(`  ✗ Tokenizer type ${tokenizerType} not set correctly`));
      allTokenizersWork = false;
      failed++;
    }
  }

  if (allTokenizersWork) {
    console.log(chalk.green(`✓ All tokenizer types work correctly (${tokenizers.length} types)`));
    passed++;
  }
} catch (error) {
  console.log(chalk.red('✗ Failed to test tokenizer types'));
  console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
  failed++;
}

// Test 6: Verify comments are present
console.log(chalk.cyan('\nTest 6: Verify inline comments...'));
try {
  const config: ProjectConfig = {
    projectName: 'test-project',
    template: 'tiny',
    tokenizer: 'bpe',
    plugins: [],
    skipInstall: false
  };
  const template = templateManager.getTemplate('tiny');
  const configContent = configGenerator.generateConfig(config, template);

  const commentChecks = [
    { check: configContent.includes('// Architecture type'), desc: 'Model type comment' },
    { check: configContent.includes('// Training batch size'), desc: 'Batch size comment' },
    { check: configContent.includes('// Learning rate'), desc: 'Learning rate comment' },
    { check: configContent.includes('// Tokenizer type'), desc: 'Tokenizer type comment' },
    { check: configContent.includes('// Maximum checkpoints to keep'), desc: 'Checkpoint comment' },
    { check: configContent.includes('// Enable TensorBoard'), desc: 'TensorBoard comment' }
  ];

  let allComments = true;
  for (const { check, desc } of commentChecks) {
    if (!check) {
      console.log(chalk.red(`  ✗ Missing comment: ${desc}`));
      allComments = false;
      failed++;
    }
  }

  if (allComments) {
    console.log(chalk.green(`✓ All inline comments present (${commentChecks.length} comments)`));
    passed++;
  }
} catch (error) {
  console.log(chalk.red('✗ Failed to verify comments'));
  console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
  failed++;
}

// Summary
console.log(chalk.yellow(`\n\nResults: ${passed} passed, ${failed} failed`));

if (failed === 0) {
  console.log(chalk.green.bold('\n✅ All config generator tests passed!\n'));
  process.exit(0);
} else {
  console.log(chalk.red.bold('\n❌ Some tests failed!\n'));
  process.exit(1);
}
