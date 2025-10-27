# Development Guide

## Project Overview

create-llm is a CLI tool for scaffolding LLM training projects. It generates production-ready Python training infrastructure from TypeScript templates.

## Setup

### Prerequisites
- Node.js 18.0.0 or higher
- npm 8.0.0 or higher
- Git

### Installation

```bash
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm
npm install
```

### Build

```bash
npm run build
```

### Development Mode

```bash
npm run dev
```

### Test Locally

```bash
node dist/index.js test-project --template nano
```

### Run Tests

```bash
npm test
```

## Project Structure

```
create-llm/
├── src/                    # TypeScript source code
│   ├── index.ts           # CLI entry point
│   ├── prompts.ts         # Interactive prompts
│   ├── scaffolder.ts      # Main scaffolding engine
│   └── *-templates.ts     # Python code templates
├── templates/             # Configuration templates
│   ├── nano.json
│   ├── tiny.json
│   ├── small.json
│   └── base.json
├── dist/                  # Compiled JavaScript
└── tests/                 # Test files
```

## Architecture

### CLI Layer
- Handles user input and validation
- Manages interactive prompts
- Orchestrates project generation

### Template System
- Generates Python code from TypeScript templates
- Injects configuration values
- Ensures code consistency

### Configuration System
- Template-based model configurations
- Hardware-aware defaults
- Extensible plugin system

## Technical Decisions

### Why TypeScript for CLI?
- Fast startup time
- Easy npm distribution
- Type safety
- Industry standard for CLI tools

### Why Template Strings?
- Simple and maintainable
- Self-contained (no external template files)
- Fast (no file I/O)
- Common pattern (used by create-react-app, etc.)

### Why Not Python for Everything?
- Slower CLI startup
- More complex distribution
- npm ecosystem is better for CLI tools

## Development Workflow

### Making Changes

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes

3. Build and test
   ```bash
   npm run build
   npm test
   ```

4. Test locally
   ```bash
   node dist/index.js test-project
   ```

5. Commit with conventional commits
   ```bash
   git commit -m "feat: add new feature"
   ```

6. Push and create PR
   ```bash
   git push origin feature/your-feature
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Testing

Run tests before committing:

```bash
npm test
```

Test the generated projects:

```bash
node dist/index.js test-project --template nano
cd test-project
pip install -r requirements.txt
python training/train.py --help
```

## Publishing

### Version Bump

```bash
npm version patch  # for bug fixes
npm version minor  # for new features
npm version major  # for breaking changes
```

### Publish to npm

```bash
npm publish
```

## Known Issues

### Technical Debt
- Some template files are large (>1000 lines)
- Could benefit from refactoring to separate template files
- Test coverage could be improved

### Future Improvements
- Separate template files (.py.template)
- Add more model architectures
- Distributed training support
- Better plugin system

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Resources

- [TypeScript Documentation](https://www.typescriptlang.org/docs/)
- [Commander.js](https://github.com/tj/commander.js)
- [Inquirer.js](https://github.com/SBoudrias/Inquirer.js)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Last Updated**: 2025-10-26
**Maintainer**: Aniket Giri
