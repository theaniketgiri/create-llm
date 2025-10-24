#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import * as path from 'path';
import * as fs from 'fs';
import { TemplateManager } from './template-manager';
import { TemplateName, Template } from './types/template';
import { CLIPrompts, ProjectConfig } from './prompts';
import { ScaffolderEngine } from './scaffolder';

/**
 * Display enhanced post-install message with next steps and guidance
 */
function displayNextSteps(config: ProjectConfig, template: Template): void {
  const { projectName, plugins } = config;
  const modelSize = (template.config.model.parameters / 1_000_000).toFixed(0);
  const templateName = config.template.toUpperCase();
  
  // Header
  console.log('\n' + chalk.green('═'.repeat(70)));
  console.log(chalk.green.bold('  ✨ Project Created Successfully! ✨'));
  console.log(chalk.green('═'.repeat(70)));
  
  // Project Details
  console.log(chalk.cyan.bold('\n📦 Project Details:'));
  console.log(chalk.gray('─'.repeat(70)));
  console.log(chalk.white(`  📁 Location:     ${chalk.bold('./' + projectName)}`));
  console.log(chalk.white(`  🎯 Template:     ${chalk.bold(templateName)} (${modelSize}M parameters)`));
  console.log(chalk.white(`  🤖 Model:        ${chalk.bold(template.config.model.type.toUpperCase())}`));
  console.log(chalk.white(`  📝 Tokenizer:    ${chalk.bold(config.tokenizer.toUpperCase())}`));
  console.log(chalk.white(`  💾 Hardware:     ${chalk.bold(template.config.hardware.recommended_gpu || 'CPU-friendly')}`));
  console.log(chalk.white(`  ⏱️  Training:     ${chalk.bold(template.config.hardware.estimated_training_time)}`));
  
  if (plugins.length > 0) {
    console.log(chalk.white(`  🔌 Plugins:      ${chalk.bold(plugins.join(', '))}`));
  }
  
  console.log(chalk.gray('─'.repeat(70)));
  
  // Quick Start
  console.log(chalk.yellow.bold('\n🚀 Quick Start:'));
  console.log(chalk.white('\n  1️⃣  Navigate to your project:'));
  console.log(chalk.cyan(`     cd ${projectName}`));
  
  console.log(chalk.white('\n  2️⃣  Install dependencies:'));
  console.log(chalk.cyan('     pip install -r requirements.txt'));
  
  console.log(chalk.white('\n  3️⃣  Prepare your data:'));
  console.log(chalk.gray('     • Place your training data in data/raw/'));
  console.log(chalk.cyan('     python tokenizer/train.py --data data/raw/sample.txt'));
  
  console.log(chalk.white('\n  4️⃣  Start training:'));
  console.log(chalk.cyan('     python training/train.py'));
  
  // Template-Specific Tips
  console.log(chalk.magenta.bold('\n💡 Template-Specific Tips:'));
  const tips = template.config.documentation.training_tips;
  tips.slice(0, 3).forEach((tip, index) => {
    console.log(chalk.gray(`  ${index + 1}. ${tip}`));
  });
  
  // Plugin-Specific Guidance
  if (plugins.length > 0) {
    console.log(chalk.blue.bold('\n🔌 Plugin Setup:'));
    
    if (plugins.includes('wandb')) {
      console.log(chalk.white('  📊 WandB (Experiment Tracking):'));
      console.log(chalk.gray('     • Login: wandb login'));
      console.log(chalk.gray('     • Configure in llm.config.js'));
      console.log(chalk.gray('     • View experiments at wandb.ai'));
    }
    
    if (plugins.includes('huggingface')) {
      console.log(chalk.white('  🤗 HuggingFace (Model Sharing):'));
      console.log(chalk.gray('     • Login: huggingface-cli login'));
      console.log(chalk.gray('     • Configure repo_id in llm.config.js'));
      console.log(chalk.gray('     • Deploy: python deploy.py --to huggingface'));
    }
    
    if (plugins.includes('synthex')) {
      console.log(chalk.white('  🎲 SynthexAI (Data Generation):'));
      console.log(chalk.gray('     • Generate data: python data/generate.py'));
      console.log(chalk.gray('     • Configure in llm.config.js'));
    }
  }
  
  // Example Workflow
  console.log(chalk.green.bold('\n📚 Example Workflow:'));
  console.log(chalk.gray('  ┌─ Prepare Data'));
  console.log(chalk.gray('  │  └─ python data/prepare.py'));
  console.log(chalk.gray('  │'));
  console.log(chalk.gray('  ├─ Train Model'));
  console.log(chalk.gray('  │  └─ python training/train.py'));
  console.log(chalk.gray('  │'));
  console.log(chalk.gray('  ├─ Evaluate'));
  console.log(chalk.gray('  │  ├─ python evaluation/evaluate.py'));
  console.log(chalk.gray('  │  └─ python evaluation/generate.py --prompt "Once upon a time"'));
  console.log(chalk.gray('  │'));
  console.log(chalk.gray('  ├─ Chat with Model'));
  console.log(chalk.gray('  │  └─ python chat.py --checkpoint checkpoints/final.pt'));
  console.log(chalk.gray('  │'));
  console.log(chalk.gray('  └─ Deploy'));
  console.log(chalk.gray('     └─ python deploy.py --to huggingface --repo-id username/model'));
  
  // Advanced Features
  console.log(chalk.cyan.bold('\n⚡ Advanced Features:'));
  console.log(chalk.white('  • Live Dashboard:    ') + chalk.gray('python training/train.py --dashboard'));
  console.log(chalk.white('  • Resume Training:   ') + chalk.gray('python training/train.py --resume checkpoints/checkpoint-1000.pt'));
  console.log(chalk.white('  • Model Comparison:  ') + chalk.gray('python compare.py model1/ model2/'));
  console.log(chalk.white('  • Custom Config:     ') + chalk.gray('Edit llm.config.js'));
  
  // Documentation Links
  console.log(chalk.yellow.bold('\n📖 Documentation & Resources:'));
  console.log(chalk.white('  • README:            ') + chalk.gray(`./${projectName}/README.md`));
  console.log(chalk.white('  • Config Guide:      ') + chalk.gray('llm.config.js (with inline comments)'));
  console.log(chalk.white('  • Plugin Docs:       ') + chalk.gray('plugins/README.md'));
  console.log(chalk.white('  • GitHub:            ') + chalk.blue.underline('https://github.com/theaniketgiri/create-llm'));
  
  // Hardware Requirements
  if (template.config.hardware.min_ram || template.config.hardware.recommended_gpu) {
    console.log(chalk.red.bold('\n⚠️  Hardware Requirements:'));
    if (template.config.hardware.min_ram) {
      console.log(chalk.gray(`  • Minimum RAM:       ${template.config.hardware.min_ram}`));
    }
    if (template.config.hardware.recommended_gpu) {
      console.log(chalk.gray(`  • Recommended GPU:   ${template.config.hardware.recommended_gpu}`));
    }
    if (!template.config.hardware.can_run_on_cpu) {
      console.log(chalk.yellow('  ⚠️  GPU required for this template'));
    } else {
      console.log(chalk.green('  ✓ Can run on CPU (slower)'));
    }
  }
  
  // Footer
  console.log('\n' + chalk.green('═'.repeat(70)));
  console.log(chalk.green.bold('  🎉 Ready to train your LLM! Good luck! 🚀'));
  console.log(chalk.green('═'.repeat(70)) + '\n');
  
  // Final tip
  console.log(chalk.gray('  💬 Need help? Check the README or open an issue on GitHub\n'));
}

const program = new Command();
const templateManager = new TemplateManager();
const prompts = new CLIPrompts(templateManager);

program
  .name('create-llm')
  .description('CLI tool to scaffold LLM training projects')
  .version('0.1.0')
  .argument('[project-name]', 'Name of the project to create')
  .option('-t, --template <template>', 'Template to use (tiny, small, base, custom)')
  .option('--tokenizer <type>', 'Tokenizer type (bpe, wordpiece, unigram)')
  .option('--skip-install', 'Skip dependency installation')
  .option('-y, --yes', 'Skip confirmation prompt')
  .action(async (projectName: string | undefined, options) => {
    console.log(chalk.blue.bold('\n🚀 Welcome to create-llm!\n'));

    try {
      // Validate template if provided
      if (options.template && !templateManager.hasTemplate(options.template)) {
        console.error(chalk.red(`\n❌ Invalid template: ${options.template}`));
        console.log(chalk.yellow(`Available templates: ${templateManager.getAvailableTemplates().join(', ')}`));
        process.exit(1);
      }

      // Validate tokenizer if provided
      if (options.tokenizer && !['bpe', 'wordpiece', 'unigram'].includes(options.tokenizer)) {
        console.error(chalk.red(`\n❌ Invalid tokenizer: ${options.tokenizer}`));
        console.log(chalk.yellow('Available tokenizers: bpe, wordpiece, unigram'));
        process.exit(1);
      }

      // Validate project name if provided
      if (projectName && !prompts.validateProjectName(projectName)) {
        console.error(chalk.red(`\n❌ Invalid project name: ${projectName}`));
        console.log(chalk.yellow('Project name must contain only lowercase letters, numbers, and hyphens'));
        process.exit(1);
      }

      // Run interactive flow
      const config = await prompts.runInteractiveFlow(
        projectName,
        options.template as TemplateName,
        options.tokenizer,
        options.skipInstall
      );

      if (!config) {
        process.exit(0);
      }

      // Get template
      const template = templateManager.getTemplate(config.template);

      // Check if directory already exists
      const projectPath = path.join(process.cwd(), config.projectName);
      if (fs.existsSync(projectPath)) {
        console.error(chalk.red(`\n❌ Directory "${config.projectName}" already exists`));
        console.log(chalk.yellow('Please choose a different project name or remove the existing directory'));
        process.exit(1);
      }

      // Create scaffolder
      const scaffolder = new ScaffolderEngine(projectPath);

      // Create project structure
      await scaffolder.createProjectStructure(config, template);

      // Copy template files
      await scaffolder.copyTemplateFiles(config, template);

      // Display enhanced post-install message
      displayNextSteps(config, template);
    } catch (error) {
      console.error(chalk.red('\n❌ Error:'), error instanceof Error ? error.message : String(error));
      process.exit(1);
    }
  });

program.parse();
