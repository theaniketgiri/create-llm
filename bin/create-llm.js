#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const inquirer = require('inquirer');
const ora = require('ora');
const fs = require('fs-extra');
const path = require('path');
const { generateProject } = require('../src/generator');

const program = new Command();

program
  .name('create-llm')
  .description('Create a custom LLM project from scratch - like create-react-app for language models')
  .version('1.0.0')
  .argument('[project-name]', 'Name of the project to create')
  .option('-t, --template <template>', 'Template to use (gpt, mistral, rwkv)', 'gpt')
  .option('-y, --yes', 'Skip prompts and use defaults')
  .option('--typescript', 'Use TypeScript instead of JavaScript')
  .action(async (projectName, options) => {
    try {
      console.log(chalk.blue.bold('\n🚀 Create LLM - Build Your Custom Language Model\n'));
      
      let answers = {};
      
      if (!projectName) {
        const nameAnswer = await inquirer.prompt([
          {
            type: 'input',
            name: 'projectName',
            message: 'What is your project named?',
            default: 'my-llm',
            validate: (input) => {
              if (!input.trim()) return 'Project name is required';
              if (!/^[a-z0-9-]+$/.test(input)) {
                return 'Project name can only contain lowercase letters, numbers, and hyphens';
              }
              return true;
            }
          }
        ]);
        projectName = nameAnswer.projectName;
      }

      if (!options.yes) {
        answers = await inquirer.prompt([
          {
            type: 'list',
            name: 'template',
            message: 'Which model architecture would you like to use?',
            choices: [
              { name: 'GPT-2 Style Transformer', value: 'gpt' },
              { name: 'Mistral (7B style)', value: 'mistral' },
              { name: 'RWKV (RNN-style)', value: 'rwkv' },
              { name: 'Mixtral (MoE)', value: 'mixtral' }
            ],
            default: options.template
          },
          {
            type: 'list',
            name: 'tokenizer',
            message: 'Which tokenizer would you like to use?',
            choices: [
              { name: 'BPE (Byte Pair Encoding)', value: 'bpe' },
              { name: 'WordPiece', value: 'wordpiece' },
              { name: 'Unigram', value: 'unigram' }
            ],
            default: 'bpe'
          },
          {
            type: 'list',
            name: 'dataset',
            message: 'Which dataset would you like to use for training?',
            choices: [
              { name: 'WikiText-103', value: 'wikitext' },
              { name: 'C4 (Common Crawl)', value: 'c4' },
              { name: 'OpenWebText', value: 'openwebtext' },
              { name: 'Custom (you provide)', value: 'custom' }
            ],
            default: 'wikitext'
          },
          {
            type: 'confirm',
            name: 'useTypescript',
            message: 'Would you like to use TypeScript?',
            default: options.typescript || false
          },
          {
            type: 'confirm',
            name: 'includeSyntheticData',
            message: 'Would you like to include synthetic data generation capabilities?',
            default: true
          }
        ]);
      } else {
        answers = {
          template: options.template,
          tokenizer: 'bpe',
          dataset: 'wikitext',
          useTypescript: options.typescript || false,
          includeSyntheticData: true
        };
      }

      const spinner = ora('Creating your LLM project...').start();
      
      await generateProject(projectName, answers);
      
      spinner.succeed(chalk.green('Project created successfully!'));
      
      console.log(chalk.cyan('\n📁 Your project structure:'));
      console.log(chalk.gray(`  ${projectName}/`));
      console.log(chalk.gray('  ├── model/               # Transformer architecture'));
      console.log(chalk.gray('  ├── tokenizer/           # Tokenizer scripts'));
      console.log(chalk.gray('  ├── data/                # Dataset preprocessing'));
      console.log(chalk.gray('  ├── training/            # Training pipeline'));
      console.log(chalk.gray('  ├── eval/                # Evaluation scripts'));
      console.log(chalk.gray('  ├── checkpoints/         # Model checkpoints'));
      console.log(chalk.gray('  ├── logs/                # Training logs'));
      console.log(chalk.gray('  └── README.md            # Project documentation'));
      
      console.log(chalk.cyan('\n🚀 Next steps:'));
      console.log(chalk.white(`  cd ${projectName}`));
      console.log(chalk.white('  python -m venv venv'));
      console.log(chalk.white('  source venv/bin/activate  # On Windows: venv\\Scripts\\activate'));
      console.log(chalk.white('  pip install -r requirements.txt'));
      console.log(chalk.white('  python tokenizer/train_tokenizer.py --input data/raw.txt'));
      console.log(chalk.white('  python data/prepare_dataset.py'));
      console.log(chalk.white('  python training/train.py --config training/config.yaml'));
      
      console.log(chalk.cyan('\n📚 Documentation:'));
      console.log(chalk.white('  Read the README.md file for detailed instructions'));
      console.log(chalk.white('  Visit: https://github.com/theaniketgiri/create-llm'));
      
      console.log(chalk.green('\n✨ Happy training!'));
      
    } catch (error) {
      console.error(chalk.red('\n❌ Error creating project:'), error.message);
      process.exit(1);
    }
  });

program.parse(); 