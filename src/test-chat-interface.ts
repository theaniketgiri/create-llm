#!/usr/bin/env node

/**
 * Test script for chat interface generation
 * Verifies that the generated chat.py has all required features
 */

import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { ScaffolderEngine } from './scaffolder';
import { TemplateManager } from './template-manager';
import { ProjectConfig } from './prompts';

console.log(chalk.blue.bold('\n🧪 Testing Chat Interface Generation\n'));

const templateManager = new TemplateManager();
const testProjectName = 'test-chat-interface';
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
    console.log(chalk.cyan('Test 1: Generate chat.py file...'));
    
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

    // Read chat.py
    const chatPath = path.join(testProjectPath, 'chat.py');
    testCheck(fs.existsSync(chatPath), 'chat.py file exists');
    
    const chatContent = fs.readFileSync(chatPath, 'utf-8');

    console.log(chalk.cyan('\nTest 2: Verify ChatSession class...'));
    
    // Check ChatSession class exists
    testCheck(
      chatContent.includes('class ChatSession:'),
      'ChatSession class defined'
    );
    
    // Check __init__ method
    testCheck(
      chatContent.includes('def __init__(') && 
      chatContent.includes('self.model') &&
      chatContent.includes('self.tokenizer') &&
      chatContent.includes('self.device'),
      'ChatSession.__init__ with model, tokenizer, device'
    );
    
    // Check context maintenance
    testCheck(
      chatContent.includes('self.context = []'),
      'Context list initialized'
    );
    
    testCheck(
      chatContent.includes('self.context_window'),
      'Context window parameter'
    );

    console.log(chalk.cyan('\nTest 3: Verify response generation...'));
    
    // Check generate_response method
    testCheck(
      chatContent.includes('def generate_response('),
      'generate_response method defined'
    );
    
    // Check generation parameters
    testCheck(
      chatContent.includes('temperature:') &&
      chatContent.includes('top_k:') &&
      chatContent.includes('top_p:'),
      'Sampling parameters (temperature, top_k, top_p)'
    );
    
    // Check context management
    testCheck(
      chatContent.includes('self.context.append(') &&
      chatContent.includes('User:') &&
      chatContent.includes('Assistant:'),
      'Context append with User/Assistant markers'
    );
    
    // Check context trimming
    testCheck(
      chatContent.includes('if len(self.context) >') &&
      chatContent.includes('self.context = self.context['),
      'Context trimming to maintain window'
    );

    console.log(chalk.cyan('\nTest 4: Verify generation logic...'));
    
    // Check _generate method
    testCheck(
      chatContent.includes('def _generate('),
      '_generate method defined'
    );
    
    // Check top-k sampling
    testCheck(
      chatContent.includes('if top_k > 0:') &&
      chatContent.includes('torch.topk'),
      'Top-k sampling implementation'
    );
    
    // Check top-p sampling
    testCheck(
      chatContent.includes('if top_p < 1.0:') &&
      chatContent.includes('cumulative_probs'),
      'Top-p (nucleus) sampling implementation'
    );
    
    // Check autoregressive generation
    testCheck(
      chatContent.includes('for _ in range(max_length):') &&
      chatContent.includes('torch.cat([input_ids'),
      'Autoregressive token generation'
    );

    console.log(chalk.cyan('\nTest 5: Verify CLI arguments...'));
    
    // Check argparse
    testCheck(
      chatContent.includes('import argparse'),
      'argparse imported'
    );
    
    testCheck(
      chatContent.includes('def parse_args():'),
      'parse_args function defined'
    );
    
    // Check required arguments
    testCheck(
      chatContent.includes('--checkpoint') &&
      chatContent.includes('required=True'),
      '--checkpoint argument (required)'
    );
    
    // Check optional arguments
    const optionalArgs = [
      '--temperature',
      '--max-length',
      '--top-k',
      '--top-p',
      '--device',
      '--context-window'
    ];
    
    for (const arg of optionalArgs) {
      testCheck(
        chatContent.includes(arg),
        `${arg} argument`
      );
    }

    console.log(chalk.cyan('\nTest 6: Verify help documentation...'));
    
    // Check help text
    testCheck(
      chatContent.includes('epilog=') &&
      chatContent.includes('Commands during chat:'),
      'Help epilog with commands'
    );
    
    testCheck(
      chatContent.includes('exit, quit') &&
      chatContent.includes('clear, reset') &&
      chatContent.includes('help'),
      'Command documentation (exit, clear, help)'
    );
    
    testCheck(
      chatContent.includes('Examples:'),
      'Usage examples in help'
    );

    console.log(chalk.cyan('\nTest 7: Verify interactive loop...'));
    
    // Check main function
    testCheck(
      chatContent.includes('def main():'),
      'main function defined'
    );
    
    // Check model loading
    testCheck(
      chatContent.includes('torch.load(') &&
      chatContent.includes('checkpoint'),
      'Checkpoint loading'
    );
    
    testCheck(
      chatContent.includes('load_model_from_config('),
      'Model loading from config'
    );
    
    testCheck(
      chatContent.includes('model.load_state_dict(checkpoint[\'model_state_dict\'])'),
      'Model state dict loading'
    );
    
    // Check tokenizer loading
    testCheck(
      chatContent.includes('Tokenizer.from_file'),
      'Tokenizer loading'
    );
    
    testCheck(
      chatContent.includes('tokenizer/tokenizer.json'),
      'Tokenizer path check'
    );

    console.log(chalk.cyan('\nTest 8: Verify chat loop...'));
    
    // Check while loop
    testCheck(
      chatContent.includes('while True:'),
      'Infinite chat loop'
    );
    
    // Check user input
    testCheck(
      chatContent.includes('input(') &&
      chatContent.includes('You:'),
      'User input prompt'
    );
    
    // Check exit commands
    testCheck(
      chatContent.includes('if user_input.lower() in [\'exit\', \'quit\'') &&
      chatContent.includes('break'),
      'Exit command handling'
    );
    
    // Check clear command
    testCheck(
      chatContent.includes('if user_input.lower() in [\'clear\', \'reset\']') &&
      chatContent.includes('reset_context()'),
      'Clear/reset command handling'
    );
    
    // Check help command
    testCheck(
      chatContent.includes('if user_input.lower() == \'help\''),
      'Help command handling'
    );

    console.log(chalk.cyan('\nTest 9: Verify loading indicator...'));
    
    // Check loading indicator
    testCheck(
      chatContent.includes('⏳') ||
      chatContent.includes('Thinking'),
      'Loading indicator during generation'
    );
    
    testCheck(
      chatContent.includes('flush=True'),
      'Flush output for loading indicator'
    );

    console.log(chalk.cyan('\nTest 10: Verify error handling...'));
    
    // Check KeyboardInterrupt
    testCheck(
      chatContent.includes('except KeyboardInterrupt:'),
      'KeyboardInterrupt handling'
    );
    
    // Check general exception handling
    testCheck(
      chatContent.includes('except Exception as e:'),
      'General exception handling'
    );
    
    // Check checkpoint existence check
    testCheck(
      chatContent.includes('if not Path(args.checkpoint).exists():'),
      'Checkpoint existence validation'
    );
    
    // Check tokenizer existence check
    testCheck(
      chatContent.includes('if not tokenizer_path.exists():'),
      'Tokenizer existence validation'
    );

    console.log(chalk.cyan('\nTest 11: Verify context management features...'));
    
    // Check reset_context method
    testCheck(
      chatContent.includes('def reset_context(self):'),
      'reset_context method'
    );
    
    // Check get_context_length method
    testCheck(
      chatContent.includes('def get_context_length('),
      'get_context_length method'
    );
    
    // Check context info display
    testCheck(
      chatContent.includes('Context:') &&
      chatContent.includes('tokens'),
      'Context length display'
    );

    console.log(chalk.cyan('\nTest 12: Verify device handling...'));
    
    // Check device selection
    testCheck(
      chatContent.includes('if args.device == \'auto\':') &&
      chatContent.includes('torch.cuda.is_available()'),
      'Auto device selection'
    );
    
    testCheck(
      chatContent.includes('model.to(device)'),
      'Model moved to device'
    );
    
    testCheck(
      chatContent.includes('device=self.device'),
      'Tensors created on correct device'
    );

    console.log(chalk.cyan('\nTest 13: Verify response cleaning...'));
    
    // Check response cleanup
    testCheck(
      chatContent.includes('response.split("User:")[0]') ||
      chatContent.includes('response.split(\'User:\')[0]'),
      'Stop at User: marker'
    );
    
    testCheck(
      chatContent.includes('response.split("Assistant:")[0]') ||
      chatContent.includes('response.split(\'Assistant:\')[0]'),
      'Stop at Assistant: marker'
    );
    
    testCheck(
      chatContent.includes('.strip()'),
      'Response trimming'
    );

    console.log(chalk.cyan('\nTest 14: Verify user experience features...'));
    
    // Check colored output
    testCheck(
      chatContent.includes('\\033[1;36m') || chatContent.includes('\\033[1;32m'),
      'ANSI color codes for better UX'
    );
    
    // Check header/separator
    testCheck(
      chatContent.includes('"=" * 60') || chatContent.includes('"=" * 70') || 
      chatContent.includes('\'=\' * 60') || chatContent.includes('\'=\' * 70'),
      'Visual separators'
    );
    
    // Check emojis
    testCheck(
      chatContent.includes('💬') || chatContent.includes('🤖') || chatContent.includes('👋'),
      'Emojis for visual appeal'
    );
    
    // Check informative messages
    testCheck(
      chatContent.includes('Device:') &&
      chatContent.includes('Loading checkpoint:') &&
      chatContent.includes('Loading model'),
      'Informative status messages'
    );

    console.log(chalk.cyan('\nTest 15: Verify model evaluation mode...'));
    
    // Check eval mode
    testCheck(
      chatContent.includes('self.model.eval()'),
      'Model set to evaluation mode'
    );
    
    // Check no_grad context
    testCheck(
      chatContent.includes('with torch.no_grad():'),
      'torch.no_grad() for inference'
    );

    console.log(chalk.cyan('\nTest 16: Verify configuration integration...'));
    
    // Check config loading
    testCheck(
      chatContent.includes('ConfigLoader'),
      'ConfigLoader imported'
    );
    
    testCheck(
      chatContent.includes('config.get(\'model.max_length\''),
      'Config used for max_length'
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
