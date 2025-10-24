#!/usr/bin/env node

/**
 * Test template validation rules
 */

import * as fs from 'fs';
import * as path from 'path';
import { TemplateManager, TemplateValidationError } from './template-manager';
import { Template } from './types/template';
import chalk from 'chalk';

console.log(chalk.blue.bold('\n🧪 Testing Template Validation\n'));

const testCases = [
  {
    name: 'Invalid model type',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'invalid', parameters: 1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  },
  {
    name: 'Negative parameters',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'gpt', parameters: -1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  },
  {
    name: 'Dimension not divisible by heads',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'gpt', parameters: 1000000, layers: 6, heads: 7, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  },
  {
    name: 'Invalid dropout',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'gpt', parameters: 1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 1.5 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  },
  {
    name: 'Invalid optimizer',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'gpt', parameters: 1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'invalid', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  },
  {
    name: 'Stride greater than max_length',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'gpt', parameters: 1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 1024, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  },
  {
    name: 'Invalid tokenizer type',
    template: {
      name: 'tiny',
      config: {
        model: { type: 'gpt', parameters: 1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
        training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
        data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
        tokenizer: { type: 'invalid', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
        hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
        documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
      }
    },
    shouldFail: true
  }
];

let passed = 0;
let failed = 0;

// Create a temporary templates directory for testing
const tempDir = path.join(__dirname, '..', 'temp-templates');
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir);
}

// Create dummy files for all template names so loading doesn't fail
const dummyTemplate = {
  name: 'tiny',
  config: {
    model: { type: 'gpt', parameters: 1000000, layers: 6, heads: 6, dim: 384, vocab_size: 32000, max_length: 512, dropout: 0.1 },
    training: { batch_size: 16, learning_rate: 0.0006, warmup_steps: 500, max_steps: 10000, eval_interval: 500, save_interval: 2000, optimizer: 'adamw', weight_decay: 0.01, gradient_clip: 1.0, mixed_precision: false, gradient_accumulation_steps: 1 },
    data: { max_length: 512, stride: 256, val_split: 0.1, shuffle: true },
    tokenizer: { type: 'bpe', vocab_size: 32000, min_frequency: 2, special_tokens: ['<pad>'] },
    hardware: { min_ram: '4GB', recommended_gpu: 'None', estimated_training_time: '10 min', can_run_on_cpu: true },
    documentation: { description: 'Test', use_cases: ['test'], hardware_notes: 'test', training_tips: ['test'] }
  }
};

for (const name of ['tiny', 'small', 'base', 'custom']) {
  const templateCopy = JSON.parse(JSON.stringify(dummyTemplate));
  templateCopy.name = name;
  fs.writeFileSync(path.join(tempDir, `${name}.json`), JSON.stringify(templateCopy, null, 2));
}

for (const testCase of testCases) {
  try {
    // Write test template to file, overwriting tiny.json
    const testPath = path.join(tempDir, 'tiny.json');
    fs.writeFileSync(testPath, JSON.stringify(testCase.template, null, 2));

    // Try to load and validate
    const manager = new TemplateManager(tempDir);
    
    if (testCase.shouldFail) {
      console.log(chalk.red(`✗ ${testCase.name}: Should have failed but passed`));
      failed++;
    } else {
      console.log(chalk.green(`✓ ${testCase.name}: Passed as expected`));
      passed++;
    }
  } catch (error) {
    if (testCase.shouldFail) {
      console.log(chalk.green(`✓ ${testCase.name}: Failed as expected`));
      console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
      passed++;
    } else {
      console.log(chalk.red(`✗ ${testCase.name}: Should have passed but failed`));
      console.log(chalk.gray(`  Error: ${error instanceof Error ? error.message : String(error)}`));
      failed++;
    }
  }
}

// Clean up
if (fs.existsSync(tempDir)) {
  fs.rmSync(tempDir, { recursive: true, force: true });
}

console.log(chalk.yellow(`\n\nResults: ${passed} passed, ${failed} failed`));

if (failed === 0) {
  console.log(chalk.green.bold('\n✅ All validation tests passed!\n'));
  process.exit(0);
} else {
  console.log(chalk.red.bold('\n❌ Some validation tests failed!\n'));
  process.exit(1);
}
