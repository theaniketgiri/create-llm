import inquirer from 'inquirer';
import chalk from 'chalk';
import { TemplateManager } from './template-manager';
import { TemplateName } from './types/template';

export interface ProjectConfig {
  projectName: string;
  template: TemplateName;
  tokenizer: 'bpe' | 'wordpiece' | 'unigram';
  plugins: string[];
  skipInstall: boolean;
}

export class CLIPrompts {
  private templateManager: TemplateManager;

  constructor(templateManager: TemplateManager) {
    this.templateManager = templateManager;
  }

  /**
   * Prompt for project name with validation
   */
  async promptProjectName(defaultName?: string): Promise<string> {
    const { projectName } = await inquirer.prompt([
      {
        type: 'input',
        name: 'projectName',
        message: 'What is your project name?',
        default: defaultName || 'my-llm',
        validate: (input: string) => {
          if (!input || input.trim() === '') {
            return 'Project name cannot be empty';
          }
          if (!/^[a-z0-9-]+$/.test(input)) {
            return 'Project name must contain only lowercase letters, numbers, and hyphens';
          }
          if (input.length < 2) {
            return 'Project name must be at least 2 characters long';
          }
          if (input.length > 50) {
            return 'Project name must be less than 50 characters';
          }
          return true;
        }
      }
    ]);
    return projectName;
  }

  /**
   * Prompt for template selection with descriptions
   */
  async promptTemplate(defaultTemplate?: TemplateName): Promise<TemplateName> {
    const templates = this.templateManager.getAllTemplates();
    
    const choices = templates.map(template => {
      const params = (template.config.model.parameters / 1_000_000).toFixed(0);
      const hardware = template.config.hardware.can_run_on_cpu ? 'CPU' : template.config.hardware.recommended_gpu;
      const time = template.config.hardware.estimated_training_time;
      
      return {
        name: `${chalk.bold(template.name.toUpperCase())} - ${template.config.documentation.description}\n  ${chalk.gray(`${params}M params | ${hardware} | ${time}`)}`,
        value: template.name,
        short: template.name
      };
    });

    const { template } = await inquirer.prompt([
      {
        type: 'list',
        name: 'template',
        message: 'Select a template:',
        choices,
        default: defaultTemplate || 'small',
        pageSize: 10
      }
    ]);

    return template as TemplateName;
  }

  /**
   * Prompt for tokenizer type selection
   */
  async promptTokenizer(defaultTokenizer?: string): Promise<'bpe' | 'wordpiece' | 'unigram'> {
    const { tokenizer } = await inquirer.prompt([
      {
        type: 'list',
        name: 'tokenizer',
        message: 'Select tokenizer type:',
        choices: [
          {
            name: `${chalk.bold('BPE')} (Byte Pair Encoding) - ${chalk.gray('Used by GPT-2, GPT-3, RoBERTa')}`,
            value: 'bpe',
            short: 'BPE'
          },
          {
            name: `${chalk.bold('WordPiece')} - ${chalk.gray('Used by BERT, DistilBERT')}`,
            value: 'wordpiece',
            short: 'WordPiece'
          },
          {
            name: `${chalk.bold('Unigram')} - ${chalk.gray('Used by T5, ALBERT')}`,
            value: 'unigram',
            short: 'Unigram'
          }
        ],
        default: defaultTokenizer || 'bpe'
      }
    ]);

    return tokenizer;
  }

  /**
   * Prompt for optional plugins selection
   */
  async promptPlugins(): Promise<string[]> {
    const { plugins } = await inquirer.prompt([
      {
        type: 'checkbox',
        name: 'plugins',
        message: 'Select optional plugins (use space to select):',
        choices: [
          {
            name: `${chalk.bold('WandB')} - ${chalk.gray('Weights & Biases integration for experiment tracking')}`,
            value: 'wandb',
            checked: false
          },
          {
            name: `${chalk.bold('HuggingFace')} - ${chalk.gray('Easy model sharing and deployment')}`,
            value: 'huggingface',
            checked: false
          }
        ]
      }
    ]);

    return plugins;
  }

  /**
   * Display confirmation prompt with all selected options
   */
  async promptConfirmation(config: ProjectConfig): Promise<boolean> {
    const template = this.templateManager.getTemplate(config.template);
    
    console.log(chalk.cyan('\n📋 Project Configuration:'));
    console.log(chalk.gray('─'.repeat(50)));
    console.log(chalk.white(`  Project Name:  ${chalk.bold(config.projectName)}`));
    console.log(chalk.white(`  Template:      ${chalk.bold(config.template.toUpperCase())}`));
    console.log(chalk.white(`  Model:         ${template.config.model.type.toUpperCase()} (${(template.config.model.parameters / 1_000_000).toFixed(0)}M parameters)`));
    console.log(chalk.white(`  Tokenizer:     ${chalk.bold(config.tokenizer.toUpperCase())}`));
    console.log(chalk.white(`  Hardware:      ${template.config.hardware.recommended_gpu}`));
    console.log(chalk.white(`  Training Time: ${template.config.hardware.estimated_training_time}`));
    
    if (config.plugins.length > 0) {
      console.log(chalk.white(`  Plugins:       ${config.plugins.map(p => chalk.bold(p)).join(', ')}`));
    } else {
      console.log(chalk.white(`  Plugins:       ${chalk.gray('None')}`));
    }
    
    console.log(chalk.gray('─'.repeat(50)));

    const { confirm } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'confirm',
        message: 'Create project with these settings?',
        default: true
      }
    ]);

    return confirm;
  }

  /**
   * Run full interactive prompt flow
   */
  async runInteractiveFlow(
    initialProjectName?: string,
    initialTemplate?: TemplateName,
    initialTokenizer?: string,
    skipInstall?: boolean
  ): Promise<ProjectConfig | null> {
    // Project name
    const projectName = initialProjectName || await this.promptProjectName();

    // Template selection
    const template = initialTemplate || await this.promptTemplate();

    // Tokenizer selection
    const tokenizer = initialTokenizer as 'bpe' | 'wordpiece' | 'unigram' || await this.promptTokenizer();

    // Plugins selection
    const plugins = await this.promptPlugins();

    // Build config
    const config: ProjectConfig = {
      projectName,
      template,
      tokenizer,
      plugins,
      skipInstall: skipInstall || false
    };

    // Confirmation
    const confirmed = await this.promptConfirmation(config);

    if (!confirmed) {
      console.log(chalk.yellow('\n❌ Project creation cancelled.\n'));
      return null;
    }

    return config;
  }

  /**
   * Validate project name
   */
  validateProjectName(name: string): boolean {
    return /^[a-z0-9-]+$/.test(name) && name.length >= 2 && name.length <= 50;
  }
}
