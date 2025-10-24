#!/usr/bin/env node

/**
 * Test script for ScaffolderEngine
 */

import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { ScaffolderEngine } from './scaffolder';
import { TemplateManager } from './template-manager';
import { ProjectConfig } from './prompts';

console.log(chalk.blue.bold('\n🧪 Testing ScaffolderEngine\n'));

const templateManager = new TemplateManager();
const testProjectName = 'test-scaffolder-project';
const testProjectPath = path.join(process.cwd(), testProjectName);

// Clean up if exists
if (fs.existsSync(testProjectPath)) {
  fs.rmSync(testProjectPath, { recursive: true, force: true });
}

const config: ProjectConfig = {
  projectName: testProjectName,
  template: 'tiny',
  tokenizer: 'bpe',
  plugins: ['wandb'],
  skipInstall: true
};

const template = templateManager.getTemplate('tiny');

async function runTests() {
  let passed = 0;
  let failed = 0;

  try {
    // Test 1: Create project structure
    console.log(chalk.cyan('Test 1: Creating project structure...'));
    const scaffolder = new ScaffolderEngine(testProjectPath);
    await scaffolder.createProjectStructure(config, template);

    if (fs.existsSync(testProjectPath)) {
      console.log(chalk.green('✓ Project directory created'));
      passed++;
    } else {
      console.log(chalk.red('✗ Project directory not created'));
      failed++;
    }

    // Test 2: Verify directories
    console.log(chalk.cyan('\nTest 2: Verifying directory structure...'));
    const requiredDirs = [
      'data',
      'data/raw',
      'data/processed',
      'models',
      'models/architectures',
      'tokenizer',
      'training',
      'training/callbacks',
      'evaluation',
      'checkpoints',
      'logs',
      'plugins'
    ];

    let allDirsExist = true;
    for (const dir of requiredDirs) {
      const dirPath = path.join(testProjectPath, dir);
      if (!fs.existsSync(dirPath)) {
        console.log(chalk.red(`✗ Missing directory: ${dir}`));
        allDirsExist = false;
        failed++;
      }
    }

    if (allDirsExist) {
      console.log(chalk.green(`✓ All ${requiredDirs.length} directories created`));
      passed++;
    }

    // Test 3: Generate files
    console.log(chalk.cyan('\nTest 3: Generating project files...'));
    await scaffolder.copyTemplateFiles(config, template);

    const requiredFiles = [
      'README.md',
      'requirements.txt',
      '.gitignore',
      'data/raw/sample.txt',
      'data/prepare.py',
      'tokenizer/train.py',
      'training/train.py',
      'evaluation/evaluate.py',
      'evaluation/generate.py',
      'chat.py',
      'deploy.py',
      'compare.py'
    ];

    let allFilesExist = true;
    for (const file of requiredFiles) {
      const filePath = path.join(testProjectPath, file);
      if (!fs.existsSync(filePath)) {
        console.log(chalk.red(`✗ Missing file: ${file}`));
        allFilesExist = false;
        failed++;
      }
    }

    if (allFilesExist) {
      console.log(chalk.green(`✓ All ${requiredFiles.length} files created`));
      passed++;
    }

    // Test 4: Verify README content
    console.log(chalk.cyan('\nTest 4: Verifying README content...'));
    const readmePath = path.join(testProjectPath, 'README.md');
    const readmeContent = fs.readFileSync(readmePath, 'utf-8');

    const readmeChecks = [
      { check: readmeContent.includes(testProjectName), desc: 'Project name' },
      { check: readmeContent.includes('TINY'), desc: 'Template name' },
      { check: readmeContent.includes('10M parameters'), desc: 'Model size' },
      { check: readmeContent.includes('BPE'), desc: 'Tokenizer type' },
      { check: readmeContent.includes('Quick Start'), desc: 'Quick start section' },
      { check: readmeContent.includes('Training Tips'), desc: 'Training tips' }
    ];

    let allReadmeChecks = true;
    for (const { check, desc } of readmeChecks) {
      if (!check) {
        console.log(chalk.red(`✗ README missing: ${desc}`));
        allReadmeChecks = false;
        failed++;
      }
    }

    if (allReadmeChecks) {
      console.log(chalk.green(`✓ README contains all required sections`));
      passed++;
    }

    // Test 5: Verify requirements.txt
    console.log(chalk.cyan('\nTest 5: Verifying requirements.txt...'));
    const reqPath = path.join(testProjectPath, 'requirements.txt');
    const reqContent = fs.readFileSync(reqPath, 'utf-8');

    const reqChecks = [
      { check: reqContent.includes('torch'), desc: 'PyTorch' },
      { check: reqContent.includes('transformers'), desc: 'Transformers' },
      { check: reqContent.includes('tokenizers'), desc: 'Tokenizers' },
      { check: reqContent.includes('tqdm'), desc: 'tqdm' }
    ];

    let allReqChecks = true;
    for (const { check, desc } of reqChecks) {
      if (!check) {
        console.log(chalk.red(`✗ requirements.txt missing: ${desc}`));
        allReqChecks = false;
        failed++;
      }
    }

    if (allReqChecks) {
      console.log(chalk.green(`✓ requirements.txt contains all core dependencies`));
      passed++;
    }

    // Test 6: Verify Python scripts have shebang
    console.log(chalk.cyan('\nTest 6: Verifying Python scripts...'));
    const pythonScripts = [
      'data/prepare.py',
      'tokenizer/train.py',
      'training/train.py',
      'evaluation/evaluate.py',
      'evaluation/generate.py',
      'chat.py',
      'deploy.py',
      'compare.py'
    ];

    let allScriptsValid = true;
    for (const script of pythonScripts) {
      const scriptPath = path.join(testProjectPath, script);
      const scriptContent = fs.readFileSync(scriptPath, 'utf-8');
      if (!scriptContent.startsWith('#!/usr/bin/env python3')) {
        console.log(chalk.red(`✗ Script missing shebang: ${script}`));
        allScriptsValid = false;
        failed++;
      }
    }

    if (allScriptsValid) {
      console.log(chalk.green(`✓ All ${pythonScripts.length} Python scripts have proper shebang`));
      passed++;
    }

    // Summary
    console.log(chalk.yellow(`\n\nResults: ${passed} passed, ${failed} failed`));

    if (failed === 0) {
      console.log(chalk.green.bold('\n✅ All scaffolder tests passed!\n'));
      return 0;
    } else {
      console.log(chalk.red.bold('\n❌ Some tests failed!\n'));
      return 1;
    }
  } catch (error) {
    console.error(chalk.red('\n❌ Test error:'), error instanceof Error ? error.message : String(error));
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
