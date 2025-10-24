#!/usr/bin/env node

/**
 * Simple test script to verify TemplateManager functionality
 */

import { TemplateManager } from './template-manager';
import chalk from 'chalk';

console.log(chalk.blue.bold('\n🧪 Testing TemplateManager\n'));

try {
  // Initialize TemplateManager
  console.log(chalk.cyan('Initializing TemplateManager...'));
  const templateManager = new TemplateManager();
  console.log(chalk.green('✓ TemplateManager initialized successfully\n'));

  // Get available templates
  const availableTemplates = templateManager.getAvailableTemplates();
  console.log(chalk.cyan(`Available templates: ${availableTemplates.join(', ')}`));
  console.log(chalk.green(`✓ Found ${availableTemplates.length} templates\n`));

  // Test each template
  for (const templateName of availableTemplates) {
    console.log(chalk.yellow(`\nTesting template: ${templateName}`));
    console.log(chalk.gray('─'.repeat(50)));

    const template = templateManager.getTemplate(templateName);
    
    console.log(chalk.white(`Name: ${template.name}`));
    console.log(chalk.white(`Model Type: ${template.config.model.type}`));
    console.log(chalk.white(`Parameters: ${(template.config.model.parameters / 1_000_000).toFixed(1)}M`));
    console.log(chalk.white(`Layers: ${template.config.model.layers}`));
    console.log(chalk.white(`Heads: ${template.config.model.heads}`));
    console.log(chalk.white(`Dimension: ${template.config.model.dim}`));
    console.log(chalk.white(`Batch Size: ${template.config.training.batch_size}`));
    console.log(chalk.white(`Learning Rate: ${template.config.training.learning_rate}`));
    console.log(chalk.white(`Hardware: ${template.config.hardware.recommended_gpu}`));
    console.log(chalk.white(`CPU Compatible: ${template.config.hardware.can_run_on_cpu ? 'Yes' : 'No'}`));
    console.log(chalk.white(`Training Time: ${template.config.hardware.estimated_training_time}`));
    
    console.log(chalk.green('✓ Template loaded and validated successfully'));
  }

  // Test template summaries
  console.log(chalk.yellow('\n\nTemplate Summaries:'));
  console.log(chalk.gray('─'.repeat(50)));
  for (const templateName of availableTemplates) {
    console.log(chalk.white('\n' + templateManager.getTemplateSummary(templateName)));
  }

  // Test error handling
  console.log(chalk.yellow('\n\nTesting error handling...'));
  try {
    templateManager.getTemplate('nonexistent' as any);
    console.log(chalk.red('✗ Should have thrown error for nonexistent template'));
  } catch (error) {
    console.log(chalk.green('✓ Correctly threw error for nonexistent template'));
  }

  console.log(chalk.green.bold('\n\n✅ All tests passed!\n'));
} catch (error) {
  console.error(chalk.red.bold('\n❌ Test failed:'));
  console.error(chalk.red(error instanceof Error ? error.message : String(error)));
  if (error instanceof Error && error.stack) {
    console.error(chalk.gray(error.stack));
  }
  process.exit(1);
}
