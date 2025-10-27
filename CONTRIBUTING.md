# Contributing to create-llm

Thanks for your interest in contributing! This document explains how the project is developed and how you can help.

## Development Philosophy

This project was built to solve a real problem: making LLM training accessible. I believe in shipping fast and iterating based on user feedback.

### About AI Assistance

I'm transparent about my development process:
- **Core logic and architecture**: Hand-written by me
- **Boilerplate and templates**: Mix of hand-written and AI-assisted
- **Documentation**: AI-assisted, then reviewed and edited by me
- **Bug fixes**: Hand-written

I use AI as a productivity tool, similar to how developers use Stack Overflow, GitHub Copilot, or code generators. The key is understanding what the code does and being able to maintain it.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm

# Install dependencies
npm install

# Build
npm run build

# Test locally
node dist/index.js test-project --template nano
```

## Project Structure

```
src/
├── index.ts                    # CLI entry point
├── prompts.ts                  # User interaction
├── config-generator.ts         # Config generation
├── template-manager.ts         # Template validation
├── scaffolder.ts               # Main scaffolding logic
└── python-*-templates.ts       # Python code templates
```

## Making Changes

### Commit Message Guidelines

Going forward, I'm following conventional commits:

```
feat: add new template
fix: resolve position embedding bug
docs: update README
chore: bump version
test: add unit tests
refactor: simplify template generation
```

### Pull Request Process

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write clear commit messages
5. Test your changes
6. Submit a PR with a clear description

## Testing

```bash
# Build and test
npm run build
npm test

# Test with different templates
node dist/index.js test-nano --template nano
node dist/index.js test-tiny --template tiny
```

## Code Style

- TypeScript with strict mode
- Descriptive variable names
- Comments for complex logic
- Keep functions focused and small

## Areas for Contribution

### High Priority
- [ ] Improve test coverage
- [ ] Add more model architectures
- [ ] Better error messages
- [ ] Performance optimizations

### Medium Priority
- [ ] Additional plugins
- [ ] More tokenizer options
- [ ] Distributed training support
- [ ] Better documentation

### Low Priority
- [ ] UI improvements
- [ ] Additional templates
- [ ] Internationalization

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email me at theaniketgiri@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Note**: This project is actively maintained. I respond to issues and PRs regularly. Don't hesitate to reach out!
