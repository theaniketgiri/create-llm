import * as fs from 'fs';
import * as path from 'path';
import { Template, TemplateName } from './types/template';

/**
 * Error thrown when template validation fails
 */
export class TemplateValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TemplateValidationError';
  }
}

/**
 * Manages loading, validation, and access to LLM training templates
 */
export class TemplateManager {
  private templates: Map<TemplateName, Template> = new Map();
  private templatesDir: string;

  constructor(templatesDir?: string) {
    // Default to templates directory relative to this file
    this.templatesDir = templatesDir || path.join(__dirname, '..', 'templates');
    this.loadTemplates();
  }

  /**
   * Load all templates from the templates directory
   */
  private loadTemplates(): void {
    const templateNames: TemplateName[] = ['nano', 'tiny', 'small', 'base', 'custom'];

    for (const name of templateNames) {
      try {
        const template = this.loadTemplate(name);
        this.validateTemplate(template);
        this.templates.set(name, template);
      } catch (error) {
        throw new Error(
          `Failed to load template "${name}": ${error instanceof Error ? error.message : String(error)}`
        );
      }
    }
  }

  /**
   * Load a single template from file
   */
  private loadTemplate(name: TemplateName): Template {
    const templatePath = path.join(this.templatesDir, `${name}.json`);

    if (!fs.existsSync(templatePath)) {
      throw new Error(`Template file not found: ${templatePath}`);
    }

    const templateContent = fs.readFileSync(templatePath, 'utf-8');
    const template = JSON.parse(templateContent) as Template;

    return template;
  }

  /**
   * Validate template structure and values
   */
  private validateTemplate(template: Template): void {
    // Validate template name
    if (!template.name || !['nano', 'tiny', 'small', 'base', 'custom'].includes(template.name)) {
      throw new TemplateValidationError(
        `Invalid template name: ${template.name}. Must be one of: nano, tiny, small, base, custom`
      );
    }

    // Validate model config
    this.validateModelConfig(template);

    // Validate training config
    this.validateTrainingConfig(template);

    // Validate data config
    this.validateDataConfig(template);

    // Validate tokenizer config
    this.validateTokenizerConfig(template);

    // Validate hardware requirements
    this.validateHardwareRequirements(template);

    // Validate documentation
    this.validateDocumentation(template);
  }

  /**
   * Validate model configuration
   */
  private validateModelConfig(template: Template): void {
    const { model } = template.config;

    if (!model) {
      throw new TemplateValidationError('Template missing model configuration');
    }

    if (!['gpt', 'bert', 't5'].includes(model.type)) {
      throw new TemplateValidationError(
        `Invalid model type: ${model.type}. Must be one of: gpt, bert, t5`
      );
    }

    if (model.parameters <= 0) {
      throw new TemplateValidationError('Model parameters must be positive');
    }

    if (model.layers <= 0) {
      throw new TemplateValidationError('Model layers must be positive');
    }

    if (model.heads <= 0) {
      throw new TemplateValidationError('Model heads must be positive');
    }

    if (model.dim <= 0) {
      throw new TemplateValidationError('Model dimension must be positive');
    }

    if (model.dim % model.heads !== 0) {
      throw new TemplateValidationError(
        `Model dimension (${model.dim}) must be divisible by number of heads (${model.heads})`
      );
    }

    if (model.vocab_size <= 0) {
      throw new TemplateValidationError('Vocabulary size must be positive');
    }

    if (model.max_length <= 0) {
      throw new TemplateValidationError('Max length must be positive');
    }

    if (model.dropout < 0 || model.dropout >= 1) {
      throw new TemplateValidationError('Dropout must be between 0 and 1');
    }
  }

  /**
   * Validate training configuration
   */
  private validateTrainingConfig(template: Template): void {
    const { training } = template.config;

    if (!training) {
      throw new TemplateValidationError('Template missing training configuration');
    }

    if (training.batch_size <= 0) {
      throw new TemplateValidationError('Batch size must be positive');
    }

    if (training.learning_rate <= 0) {
      throw new TemplateValidationError('Learning rate must be positive');
    }

    if (training.warmup_steps < 0) {
      throw new TemplateValidationError('Warmup steps must be non-negative');
    }

    if (training.max_steps <= 0) {
      throw new TemplateValidationError('Max steps must be positive');
    }

    if (training.eval_interval <= 0) {
      throw new TemplateValidationError('Eval interval must be positive');
    }

    if (training.save_interval <= 0) {
      throw new TemplateValidationError('Save interval must be positive');
    }

    if (!['adamw', 'adam', 'sgd'].includes(training.optimizer)) {
      throw new TemplateValidationError(
        `Invalid optimizer: ${training.optimizer}. Must be one of: adamw, adam, sgd`
      );
    }

    if (training.weight_decay < 0) {
      throw new TemplateValidationError('Weight decay must be non-negative');
    }

    if (training.gradient_clip <= 0) {
      throw new TemplateValidationError('Gradient clip must be positive');
    }

    if (training.gradient_accumulation_steps <= 0) {
      throw new TemplateValidationError('Gradient accumulation steps must be positive');
    }
  }

  /**
   * Validate data configuration
   */
  private validateDataConfig(template: Template): void {
    const { data } = template.config;

    if (!data) {
      throw new TemplateValidationError('Template missing data configuration');
    }

    if (data.max_length <= 0) {
      throw new TemplateValidationError('Data max length must be positive');
    }

    if (data.stride <= 0) {
      throw new TemplateValidationError('Data stride must be positive');
    }

    if (data.stride > data.max_length) {
      throw new TemplateValidationError('Data stride cannot be greater than max length');
    }

    if (data.val_split < 0 || data.val_split >= 1) {
      throw new TemplateValidationError('Validation split must be between 0 and 1');
    }
  }

  /**
   * Validate tokenizer configuration
   */
  private validateTokenizerConfig(template: Template): void {
    const { tokenizer } = template.config;

    if (!tokenizer) {
      throw new TemplateValidationError('Template missing tokenizer configuration');
    }

    if (!['bpe', 'wordpiece', 'unigram'].includes(tokenizer.type)) {
      throw new TemplateValidationError(
        `Invalid tokenizer type: ${tokenizer.type}. Must be one of: bpe, wordpiece, unigram`
      );
    }

    if (tokenizer.vocab_size <= 0) {
      throw new TemplateValidationError('Tokenizer vocab size must be positive');
    }

    if (tokenizer.min_frequency < 0) {
      throw new TemplateValidationError('Tokenizer min frequency must be non-negative');
    }

    if (!Array.isArray(tokenizer.special_tokens) || tokenizer.special_tokens.length === 0) {
      throw new TemplateValidationError('Tokenizer must have at least one special token');
    }
  }

  /**
   * Validate hardware requirements
   */
  private validateHardwareRequirements(template: Template): void {
    const { hardware } = template.config;

    if (!hardware) {
      throw new TemplateValidationError('Template missing hardware requirements');
    }

    if (!hardware.min_ram || hardware.min_ram.trim() === '') {
      throw new TemplateValidationError('Hardware min_ram is required');
    }

    if (!hardware.recommended_gpu || hardware.recommended_gpu.trim() === '') {
      throw new TemplateValidationError('Hardware recommended_gpu is required');
    }

    if (!hardware.estimated_training_time || hardware.estimated_training_time.trim() === '') {
      throw new TemplateValidationError('Hardware estimated_training_time is required');
    }

    if (typeof hardware.can_run_on_cpu !== 'boolean') {
      throw new TemplateValidationError('Hardware can_run_on_cpu must be a boolean');
    }
  }

  /**
   * Validate documentation
   */
  private validateDocumentation(template: Template): void {
    const { documentation } = template.config;

    if (!documentation) {
      throw new TemplateValidationError('Template missing documentation');
    }

    if (!documentation.description || documentation.description.trim() === '') {
      throw new TemplateValidationError('Documentation description is required');
    }

    if (!Array.isArray(documentation.use_cases) || documentation.use_cases.length === 0) {
      throw new TemplateValidationError('Documentation must have at least one use case');
    }

    if (!documentation.hardware_notes || documentation.hardware_notes.trim() === '') {
      throw new TemplateValidationError('Documentation hardware_notes is required');
    }

    if (!Array.isArray(documentation.training_tips) || documentation.training_tips.length === 0) {
      throw new TemplateValidationError('Documentation must have at least one training tip');
    }
  }

  /**
   * Get a template by name
   */
  public getTemplate(name: TemplateName): Template {
    const template = this.templates.get(name);

    if (!template) {
      throw new Error(
        `Template "${name}" not found. Available templates: ${this.getAvailableTemplates().join(', ')}`
      );
    }

    return template;
  }

  /**
   * Get all available template names
   */
  public getAvailableTemplates(): TemplateName[] {
    return Array.from(this.templates.keys());
  }

  /**
   * Check if a template exists
   */
  public hasTemplate(name: string): name is TemplateName {
    return this.templates.has(name as TemplateName);
  }

  /**
   * Get all templates
   */
  public getAllTemplates(): Template[] {
    return Array.from(this.templates.values());
  }

  /**
   * Get template summary for display
   */
  public getTemplateSummary(name: TemplateName): string {
    const template = this.getTemplate(name);
    const { model, hardware, documentation } = template.config;

    return `${name.toUpperCase()} - ${documentation.description}
  Parameters: ${(model.parameters / 1_000_000).toFixed(0)}M
  Hardware: ${hardware.recommended_gpu}
  Training Time: ${hardware.estimated_training_time}
  CPU Compatible: ${hardware.can_run_on_cpu ? 'Yes' : 'No'}`;
  }
}
