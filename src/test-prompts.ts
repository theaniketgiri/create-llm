#!/usr/bin/env node

/**
 * Test script for CLI prompts
 */

import chalk from 'chalk';
import { TemplateManager } from './template-manager';
import { CLIPrompts } from './prompts';

console.log(chalk.blue.bold('\n🧪 Testing CLI Prompts\n'));

const templateManager = new TemplateManager();
const prompts = new CLIPrompts(templateManager);

// Test project name validation
console.log(chalk.cyan('Testing project name validation:'));
const testNames = [
  { name: 'my-llm', expected: true },
  { name: 'test-project-123', expected: true },
  { name: 'a', expected: false },
  { name: 'My-Project', expected: false },
  { name: 'my_project', expected: false },
  { name: 'my project', expected: false },
  { name: '', expected: false }
];

let passed = 0;
let failed = 0;

for (const test of testNames) {
  const result = prompts.validateProjectName(test.name);
  if (result === test.expected) {
    console.log(chalk.green(`  ✓ "${test.name}" -> ${result} (expected ${test.expected})`));
    passed++;
  } else {
    console.log(chalk.red(`  ✗ "${test.name}" -> ${result} (expected ${test.expected})`));
    failed++;
  }
}

console.log(chalk.yellow(`\nResults: ${passed} passed, ${failed} failed`));

if (failed === 0) {
  console.log(chalk.green.bold('\n✅ All prompt validation tests passed!\n'));
  process.exit(0);
} else {
  console.log(chalk.red.bold('\n❌ Some tests failed!\n'));
  process.exit(1);
}
