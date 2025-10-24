/**
 * Template configuration types for create-llm
 */

export interface HardwareRequirements {
  min_ram: string;
  recommended_gpu: string;
  estimated_training_time: string;
  can_run_on_cpu: boolean;
}

export interface ModelConfig {
  type: 'gpt' | 'bert' | 't5';
  parameters: number;
  layers: number;
  heads: number;
  dim: number;
  vocab_size: number;
  max_length: number;
  dropout: number;
}

export interface TrainingConfig {
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  max_steps: number;
  eval_interval: number;
  save_interval: number;
  optimizer: 'adamw' | 'adam' | 'sgd';
  weight_decay: number;
  gradient_clip: number;
  mixed_precision: boolean;
  gradient_accumulation_steps: number;
}

export interface DataConfig {
  max_length: number;
  stride: number;
  val_split: number;
  shuffle: boolean;
}

export interface TokenizerConfig {
  type: 'bpe' | 'wordpiece' | 'unigram';
  vocab_size: number;
  min_frequency: number;
  special_tokens: string[];
}

export interface TemplateDocumentation {
  description: string;
  use_cases: string[];
  hardware_notes: string;
  training_tips: string[];
}

export interface Template {
  name: 'tiny' | 'small' | 'base' | 'custom';
  config: {
    model: ModelConfig;
    training: TrainingConfig;
    data: DataConfig;
    tokenizer: TokenizerConfig;
    hardware: HardwareRequirements;
    documentation: TemplateDocumentation;
  };
}

export type TemplateName = Template['name'];
