#!/usr/bin/env node

/**
 * Test script for plugin system generation
 * Verifies that the plugin system files are generated correctly
 */

import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { ScaffolderEngine } from './scaffolder';
import { TemplateManager } from './template-manager';
import { ProjectConfig } from './prompts';

console.log(chalk.blue.bold('\n🧪 Testing Plugin System Generation\n'));

const templateManager = new TemplateManager();
const testProjectName = 'test-plugin-system';
const testProjectPath = path.join(process.cwd(), testProjectName);

// Clean up if exists
if (fs.existsSync(testProjectPath)) {
  fs.rmSync(testProjectPath, { recursive: true, force: true });
}

let passed = 0;
let failed = 0;

function testCheck(condition: boolean, description: string): void {
  if (condition) {
    console.log(chalk.green(`  ✓ ${description}`));
    passed++;
  } else {
    console.log(chalk.red(`  ✗ ${description}`));
    failed++;
  }
}

async function runTests() {
  try {
    console.log(chalk.cyan('Test 1: Generate plugin system files...'));
    
    const config: ProjectConfig = {
      projectName: testProjectName,
      template: 'tiny',
      tokenizer: 'bpe',
      plugins: [],
      skipInstall: true
    };
    
    const template = templateManager.getTemplate('tiny');
    const scaffolder = new ScaffolderEngine(testProjectPath);
    
    await scaffolder.createProjectStructure(config, template);
    await scaffolder.copyTemplateFiles(config, template);

    // Check plugin directory exists
    const pluginsDir = path.join(testProjectPath, 'plugins');
    testCheck(fs.existsSync(pluginsDir), 'plugins/ directory exists');

    // Check plugin files exist
    const pluginFiles = [
      '__init__.py',
      'base.py',
      'plugin_manager.py',
      'example_plugin.py',
      'README.md'
    ];

    for (const file of pluginFiles) {
      const filePath = path.join(pluginsDir, file);
      testCheck(fs.existsSync(filePath), `plugins/${file} exists`);
    }

    console.log(chalk.cyan('\nTest 2: Verify Plugin base class...'));
    
    const basePath = path.join(pluginsDir, 'base.py');
    const baseContent = fs.readFileSync(basePath, 'utf-8');

    // Check Plugin class
    testCheck(
      baseContent.includes('class Plugin(ABC):'),
      'Plugin base class defined with ABC'
    );

    // Check abstract method
    testCheck(
      baseContent.includes('@abstractmethod') &&
      baseContent.includes('def initialize('),
      'initialize() is abstract method'
    );

    // Check lifecycle methods
    const lifecycleMethods = [
      'on_train_begin',
      'on_train_end',
      'on_epoch_begin',
      'on_epoch_end',
      'on_step_begin',
      'on_step_end',
      'on_validation_begin',
      'on_validation_end',
      'on_checkpoint_save',
      'cleanup'
    ];

    for (const method of lifecycleMethods) {
      testCheck(
        baseContent.includes(`def ${method}(`),
        `${method}() method defined`
      );
    }

    // Check PluginError
    testCheck(
      baseContent.includes('class PluginError(Exception):'),
      'PluginError exception class defined'
    );

    console.log(chalk.cyan('\nTest 3: Verify PluginManager class...'));
    
    const managerPath = path.join(pluginsDir, 'plugin_manager.py');
    const managerContent = fs.readFileSync(managerPath, 'utf-8');

    // Check PluginManager class
    testCheck(
      managerContent.includes('class PluginManager:'),
      'PluginManager class defined'
    );

    // Check initialization
    testCheck(
      managerContent.includes('def __init__(self, config:') &&
      managerContent.includes('self.plugins:') &&
      managerContent.includes('self.failed_plugins:'),
      'PluginManager initialization with plugins dict and failed list'
    );

    // Check _load_plugins method
    testCheck(
      managerContent.includes('def _load_plugins('),
      '_load_plugins() method defined'
    );

    // Check _load_plugin method
    testCheck(
      managerContent.includes('def _load_plugin(') &&
      managerContent.includes('importlib.import_module'),
      '_load_plugin() method with importlib'
    );

    // Check error handling
    testCheck(
      managerContent.includes('try:') &&
      managerContent.includes('except Exception as e:') &&
      managerContent.includes('self.failed_plugins.append'),
      'Graceful error handling for failed plugins'
    );

    // Check get_plugin method
    testCheck(
      managerContent.includes('def get_plugin(') &&
      managerContent.includes('return self.plugins.get('),
      'get_plugin() method'
    );

    // Check has_plugin method
    testCheck(
      managerContent.includes('def has_plugin(') &&
      managerContent.includes('return') &&
      managerContent.includes('in self.plugins'),
      'has_plugin() method'
    );

    // Check get_all_plugins method
    testCheck(
      managerContent.includes('def get_all_plugins(') &&
      managerContent.includes('return list(self.plugins.values())'),
      'get_all_plugins() method'
    );

    // Check lifecycle hook methods in manager
    const managerHooks = [
      'on_train_begin',
      'on_train_end',
      'on_epoch_begin',
      'on_epoch_end',
      'on_step_begin',
      'on_step_end',
      'on_validation_begin',
      'on_validation_end',
      'on_checkpoint_save',
      'cleanup'
    ];

    for (const hook of managerHooks) {
      testCheck(
        managerContent.includes(`def ${hook}(`),
        `PluginManager.${hook}() method`
      );
    }

    // Check that manager calls plugins in try-except
    testCheck(
      managerContent.includes('for plugin in self.plugins.values():') &&
      managerContent.includes('try:') &&
      managerContent.includes('plugin.on_'),
      'Manager calls plugin hooks with error handling'
    );

    // Check create_plugin_manager function
    testCheck(
      managerContent.includes('def create_plugin_manager('),
      'create_plugin_manager() factory function'
    );

    console.log(chalk.cyan('\nTest 4: Verify example plugin...'));
    
    const examplePath = path.join(pluginsDir, 'example_plugin.py');
    const exampleContent = fs.readFileSync(examplePath, 'utf-8');

    // Check ExamplePlugin class
    testCheck(
      exampleContent.includes('class ExamplePlugin(Plugin):'),
      'ExamplePlugin inherits from Plugin'
    );

    // Check initialize implementation
    testCheck(
      exampleContent.includes('def initialize(') &&
      exampleContent.includes('return True'),
      'initialize() implemented with return True'
    );

    // Check some lifecycle hooks are implemented
    testCheck(
      exampleContent.includes('def on_train_begin(') &&
      exampleContent.includes('def on_step_end(') &&
      exampleContent.includes('def on_train_end('),
      'Example lifecycle hooks implemented'
    );

    // Check documentation
    testCheck(
      exampleContent.includes('"""') &&
      exampleContent.includes('Example plugin'),
      'Example plugin has documentation'
    );

    console.log(chalk.cyan('\nTest 5: Verify plugins __init__.py...'));
    
    const initPath = path.join(pluginsDir, '__init__.py');
    const initContent = fs.readFileSync(initPath, 'utf-8');

    // Check imports
    testCheck(
      initContent.includes('from .base import Plugin, PluginError'),
      'Imports Plugin and PluginError from base'
    );

    testCheck(
      initContent.includes('from .plugin_manager import PluginManager, create_plugin_manager'),
      'Imports PluginManager and factory function'
    );

    // Check __all__
    testCheck(
      initContent.includes('__all__ = [') &&
      initContent.includes("'Plugin'") &&
      initContent.includes("'PluginManager'"),
      '__all__ exports defined'
    );

    console.log(chalk.cyan('\nTest 6: Verify plugins README...'));
    
    const readmePath = path.join(pluginsDir, 'README.md');
    const readmeContent = fs.readFileSync(readmePath, 'utf-8');

    // Check README sections
    testCheck(
      readmeContent.includes('# Plugins'),
      'README has main heading'
    );

    testCheck(
      readmeContent.includes('## Available Plugins') ||
      readmeContent.includes('## Built-in Plugins'),
      'README lists available plugins'
    );

    testCheck(
      readmeContent.includes('## Creating a Custom Plugin'),
      'README has custom plugin creation guide'
    );

    testCheck(
      readmeContent.includes('## Plugin Lifecycle Hooks'),
      'README documents lifecycle hooks'
    );

    testCheck(
      readmeContent.includes('wandb') &&
      readmeContent.includes('synthex') &&
      readmeContent.includes('huggingface'),
      'README mentions built-in plugins'
    );

    // Check code examples
    testCheck(
      readmeContent.includes('```python') &&
      readmeContent.includes('class') &&
      readmeContent.includes('Plugin'),
      'README has Python code examples'
    );

    testCheck(
      readmeContent.includes('```javascript') &&
      readmeContent.includes('plugins:'),
      'README has config example'
    );

    console.log(chalk.cyan('\nTest 7: Verify plugin discovery from config...'));
    
    // Check that PluginManager reads from config
    testCheck(
      managerContent.includes("self.config.get('plugins', [])"),
      'PluginManager reads plugins from config'
    );

    // Check plugin naming convention
    testCheck(
      managerContent.includes('module_name = f"{plugin_name}_plugin"'),
      'Plugin module naming convention (plugin_name_plugin.py)'
    );

    // Check plugin class naming convention
    testCheck(
      managerContent.includes("class_name = ''.join(word.capitalize()") &&
      managerContent.includes("+ 'Plugin'"),
      'Plugin class naming convention (PluginNamePlugin)'
    );

    console.log(chalk.cyan('\nTest 8: Verify error handling...'));
    
    // Check graceful degradation
    testCheck(
      managerContent.includes('Warning: Failed to load plugin') &&
      managerContent.includes('Training will continue without this plugin'),
      'Warning message for failed plugins'
    );

    // Check error tracking
    testCheck(
      managerContent.includes('self.failed_plugins.append(plugin_name)'),
      'Failed plugins are tracked'
    );

    // Check try-except in lifecycle hooks
    const hookErrorHandling = managerHooks.every(hook => {
      const hookSection = managerContent.substring(
        managerContent.indexOf(`def ${hook}(`),
        managerContent.indexOf(`def ${hook}(`) + 500
      );
      return hookSection.includes('try:') && hookSection.includes('except Exception');
    });

    testCheck(
      hookErrorHandling,
      'All lifecycle hooks have try-except error handling'
    );

    console.log(chalk.cyan('\nTest 9: Verify typing annotations...'));
    
    // Check type hints in base.py
    testCheck(
      baseContent.includes('from typing import Dict, Any, Optional'),
      'Type hints imported in base.py'
    );

    testCheck(
      baseContent.includes('config: Dict[str, Any]') &&
      baseContent.includes('-> bool:') &&
      baseContent.includes('-> None:'),
      'Type annotations used in Plugin class'
    );

    // Check type hints in plugin_manager.py
    testCheck(
      managerContent.includes('from typing import Dict, List, Optional, Any'),
      'Type hints imported in plugin_manager.py'
    );

    testCheck(
      managerContent.includes('self.plugins: Dict[str, Plugin]') &&
      managerContent.includes('self.failed_plugins: List[str]'),
      'Type annotations used in PluginManager'
    );

    console.log(chalk.cyan('\nTest 10: Verify plugin system integration...'));
    
    // Check that plugins directory is in sys.path
    testCheck(
      managerContent.includes('sys.path.insert(0, str(plugins_dir))'),
      'Plugins directory added to sys.path'
    );

    // Check plugin initialization
    testCheck(
      managerContent.includes('plugin.initialize(self.config)'),
      'Plugins are initialized with config'
    );

    // Check plugin storage
    testCheck(
      managerContent.includes('self.plugins[plugin_name] = plugin'),
      'Loaded plugins are stored in dictionary'
    );

    // Clean up
    console.log(chalk.cyan('\nCleaning up test project...'));
    fs.rmSync(testProjectPath, { recursive: true, force: true });
    console.log(chalk.green('✓ Cleanup complete'));

  } catch (error) {
    console.error(chalk.red('\n❌ Test failed with error:'), error);
    failed++;
  }
}

// Run tests
runTests().then(() => {
  console.log(chalk.blue.bold('\n📊 Test Results\n'));
  console.log(chalk.green(`✓ Passed: ${passed}`));
  console.log(chalk.red(`✗ Failed: ${failed}`));
  console.log(chalk.blue(`Total: ${passed + failed}\n`));

  if (failed > 0) {
    process.exit(1);
  }
});
