#!/usr/bin/env node

/**
 * Test script for Python file generation
 */

import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { ScaffolderEngine } from './scaffolder';
import { TemplateManager } from './template-manager';
import { ProjectConfig } from './prompts';

console.log(chalk.blue.bold('\n🧪 Testing Python File Generation\n'));

const templateManager = new TemplateManager();
const testProjectName = 'test-python-gen';
const testProjectPath = path.join(process.cwd(), testProjectName);

// Clean up if exists
if (fs.existsSync(testProjectPath)) {
  fs.rmSync(testProjectPath, { recursive: true, force: true });
}

let passed = 0;
let failed = 0;

async function runTests() {
  try {
    // Test 1: Generate files without plugins
    console.log(chalk.cyan('Test 1: Generate files without plugins...'));
    const config1: ProjectConfig = {
      projectName: testProjectName,
      template: 'tiny',
      tokenizer: 'bpe',
      plugins: [],
      skipInstall: true
    };
    const template1 = templateManager.getTemplate('tiny');
    const scaffolder1 = new ScaffolderEngine(testProjectPath);
    
    await scaffolder1.createProjectStructure(config1, template1);
    await scaffolder1.copyTemplateFiles(config1, template1);

    // Check requirements.txt
    const reqPath = path.join(testProjectPath, 'requirements.txt');
    const reqContent = fs.readFileSync(reqPath, 'utf-8');

    const reqChecks = [
      { check: reqContent.includes('torch>=2.0.0'), desc: 'PyTorch' },
      { check: reqContent.includes('transformers>=4.30.0'), desc: 'Transformers' },
      { check: reqContent.includes('tokenizers>=0.13.0'), desc: 'Tokenizers' },
      { check: reqContent.includes('tqdm>=4.65.0'), desc: 'tqdm' },
      { check: reqContent.includes('numpy>=1.24.0'), desc: 'numpy' },
      { check: reqContent.includes('datasets>=2.14.0'), desc: 'datasets' },
      { check: reqContent.includes('tensorboard>=2.13.0'), desc: 'tensorboard' },
      { check: reqContent.includes('matplotlib>=3.7.0'), desc: 'matplotlib' },
      { check: reqContent.includes('# wandb>=0.15.0'), desc: 'wandb commented' },
      { check: reqContent.includes('# huggingface-hub>=0.16.0'), desc: 'huggingface-hub commented' }
    ];

    let allReqChecks = true;
    for (const { check, desc } of reqChecks) {
      if (!check) {
        console.log(chalk.red(`  ✗ requirements.txt missing: ${desc}`));
        allReqChecks = false;
        failed++;
      }
    }

    if (allReqChecks) {
      console.log(chalk.green(`✓ requirements.txt generated correctly (${reqChecks.length} checks)`));
      passed++;
    }

    // Check .gitignore
    const gitignorePath = path.join(testProjectPath, '.gitignore');
    const gitignoreContent = fs.readFileSync(gitignorePath, 'utf-8');

    const gitignoreChecks = [
      { check: gitignoreContent.includes('__pycache__/'), desc: 'Python cache' },
      { check: gitignoreContent.includes('checkpoints/'), desc: 'Checkpoints' },
      { check: gitignoreContent.includes('logs/'), desc: 'Logs' },
      { check: gitignoreContent.includes('*.pt'), desc: 'PyTorch files' },
      { check: gitignoreContent.includes('tokenizer/tokenizer.json'), desc: 'Tokenizer' },
      { check: gitignoreContent.includes('wandb/'), desc: 'WandB' },
      { check: gitignoreContent.includes('.env'), desc: 'Environment files' },
      { check: gitignoreContent.includes('venv/'), desc: 'Virtual env' }
    ];

    let allGitignoreChecks = true;
    for (const { check, desc } of gitignoreChecks) {
      if (!check) {
        console.log(chalk.red(`  ✗ .gitignore missing: ${desc}`));
        allGitignoreChecks = false;
        failed++;
      }
    }

    if (allGitignoreChecks) {
      console.log(chalk.green(`✓ .gitignore generated correctly (${gitignoreChecks.length} checks)`));
      passed++;
    }

    // Check sample data
    const samplePath = path.join(testProjectPath, 'data/raw/sample.txt');
    const sampleContent = fs.readFileSync(samplePath, 'utf-8');

    const sampleChecks = [
      { check: sampleContent.includes('Sample Training Data'), desc: 'Title' },
      { check: sampleContent.includes('TINY Template'), desc: 'Template name' },
      { check: sampleContent.includes('10M parameters'), desc: 'Model size' },
      { check: sampleContent.includes('Data Quality Guidelines'), desc: 'Guidelines section' },
      { check: sampleContent.includes('Example Domains'), desc: 'Domains section' },
      { check: sampleContent.includes('Next Steps'), desc: 'Next steps' }
    ];

    let allSampleChecks = true;
    for (const { check, desc } of sampleChecks) {
      if (!check) {
        console.log(chalk.red(`  ✗ sample.txt missing: ${desc}`));
        allSampleChecks = false;
        failed++;
      }
    }

    if (allSampleChecks) {
      console.log(chalk.green(`✓ sample.txt generated correctly (${sampleChecks.length} checks)`));
      passed++;
    }

    // Clean up
    fs.rmSync(testProjectPath, { recursive: true, force: true });

    // Test 2: Generate files with plugins
    console.log(chalk.cyan('\nTest 2: Generate files with plugins...'));
    const config2: ProjectConfig = {
      projectName: testProjectName,
      template: 'small',
      tokenizer: 'wordpiece',
      plugins: ['wandb', 'huggingface'],
      skipInstall: true
    };
    const template2 = templateManager.getTemplate('small');
    const scaffolder2 = new ScaffolderEngine(testProjectPath);
    
    await scaffolder2.createProjectStructure(config2, template2);
    await scaffolder2.copyTemplateFiles(config2, template2);

    const reqPath2 = path.join(testProjectPath, 'requirements.txt');
    const reqContent2 = fs.readFileSync(reqPath2, 'utf-8');

    const pluginChecks = [
      { check: reqContent2.includes('wandb>=0.15.0') && !reqContent2.includes('# wandb>=0.15.0'), desc: 'WandB enabled' },
      { check: reqContent2.includes('huggingface-hub>=0.16.0') && !reqContent2.includes('# huggingface-hub>=0.16.0'), desc: 'HuggingFace enabled' },
      { check: reqContent2.includes('WandB plugin enabled'), desc: 'WandB comment' },
      { check: reqContent2.includes('HuggingFace plugin enabled'), desc: 'HuggingFace comment' }
    ];

    let allPluginChecks = true;
    for (const { check, desc } of pluginChecks) {
      if (!check) {
        console.log(chalk.red(`  ✗ Plugin dependency missing: ${desc}`));
        allPluginChecks = false;
        failed++;
      }
    }

    if (allPluginChecks) {
      console.log(chalk.green(`✓ Plugin dependencies added correctly (${pluginChecks.length} checks)`));
      passed++;
    }

    // Test 3: Verify template-specific sample data
    console.log(chalk.cyan('\nTest 3: Verify template-specific sample data...'));
    const samplePath2 = path.join(testProjectPath, 'data/raw/sample.txt');
    const sampleContent2 = fs.readFileSync(samplePath2, 'utf-8');

    const templateChecks = [
      { check: sampleContent2.includes('SMALL Template'), desc: 'Small template name' },
      { check: sampleContent2.includes('100M parameters'), desc: 'Small model size' },
      { check: sampleContent2.includes('100MB-1GB'), desc: 'Small data size' },
      { check: sampleContent2.includes('NVIDIA RTX 3060'), desc: 'Small hardware' }
    ];

    let allTemplateChecks = true;
    for (const { check, desc } of templateChecks) {
      if (!check) {
        console.log(chalk.red(`  ✗ Template-specific info missing: ${desc}`));
        allTemplateChecks = false;
        failed++;
      }
    }

    if (allTemplateChecks) {
      console.log(chalk.green(`✓ Template-specific sample data correct (${templateChecks.length} checks)`));
      passed++;
    }

    // Summary
    console.log(chalk.yellow(`\n\nResults: ${passed} passed, ${failed} failed`));

    if (failed === 0) {
      console.log(chalk.green.bold('\n✅ All Python file generation tests passed!\n'));
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
