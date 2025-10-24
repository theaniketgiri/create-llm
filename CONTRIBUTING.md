# Contributing to create-llm

Thank you for your interest in contributing to create-llm! 🎉

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 🎨 Add new templates
- 🔌 Create plugins

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/theaniketgiri/create-llm.git
cd create-llm
npm install
```

### 2. Build

```bash
npm run build
```

### 3. Test Locally

```bash
node dist/index.js test-project --template nano
```

## Development Workflow

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Build and test:
   ```bash
   npm run build
   npm test
   ```

4. Commit with clear messages:
   ```bash
   git commit -m "feat: add new feature"
   ```

5. Push and create a PR:
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
create-llm/
├── src/
│   ├── index.ts              # CLI entry point
│   ├── template-manager.ts   # Template loading
│   ├── config-generator.ts   # Config generation
│   ├── scaffolder.ts         # Project scaffolding
│   ├── prompts.ts           # Interactive prompts
│   ├── python-*.ts          # Python code templates
│   └── types/               # TypeScript types
├── templates/               # Template configurations
│   ├── nano.json
│   ├── tiny.json
│   ├── small.json
│   └── base.json
└── dist/                    # Compiled output
```

## Adding a New Template

1. Create template config in `templates/`:

```json
{
  "name": "your-template",
  "config": {
    "model": {
      "type": "gpt",
      "parameters": 1000000,
      "layers": 4,
      "heads": 4,
      "dim": 256,
      "vocab_size": 10000,
      "max_length": 512,
      "dropout": 0.1
    },
    // ... more config
  }
}
```

2. Add to `src/types/template.ts`:

```typescript
export interface Template {
  name: 'nano' | 'tiny' | 'small' | 'base' | 'your-template' | 'custom';
  // ...
}
```

3. Update `src/template-manager.ts`:

```typescript
const templateNames: TemplateName[] = ['nano', 'tiny', 'small', 'base', 'your-template', 'custom'];
```

4. Add model architecture in `src/python-templates.ts`

5. Test thoroughly!

## Code Style

- Use TypeScript strict mode
- Follow existing code patterns
- Add comments for complex logic
- Use meaningful variable names
- Keep functions focused and small

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes

Examples:
```
feat: add NANO template for beginners
fix: resolve vocab size mismatch issue
docs: improve README quick start section
```

## Testing

### Manual Testing

```bash
# Test project generation
node dist/index.js test-nano --template nano --skip-install

# Test with different templates
node dist/index.js test-tiny --template tiny
node dist/index.js test-small --template small

# Test interactive mode
node dist/index.js
```

### Automated Tests

```bash
npm test
```

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG.md**
5. **Request review** from maintainers

### PR Checklist

- [ ] Code builds without errors
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Test with minimal example

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run `npx create-llm ...`
2. ...

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- OS: [e.g. Windows 11, macOS 14, Ubuntu 22.04]
- Node.js version: [e.g. 18.0.0]
- npm version: [e.g. 8.0.0]
- create-llm version: [e.g. 1.0.0]

**Additional context**
Any other relevant information.
```

## Feature Requests

We love new ideas! Please:

1. Check if it's already requested
2. Explain the use case
3. Describe the proposed solution
4. Consider alternatives

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other relevant information.
```

## Documentation

### Improving Docs

- Fix typos and grammar
- Add examples
- Clarify confusing sections
- Add troubleshooting tips
- Improve code comments

### Documentation Structure

- `README.md` - Main documentation
- `CONTRIBUTING.md` - This file
- `docs/` - Detailed guides (future)
- Code comments - Inline documentation

## Community Guidelines

### Be Respectful

- Be kind and courteous
- Respect different viewpoints
- Accept constructive criticism
- Focus on what's best for the project

### Be Collaborative

- Help others
- Share knowledge
- Review PRs constructively
- Celebrate contributions

## Questions?

- 💬 [Discord Community](https://discord.gg/create-llm)
- 📧 [Email](mailto:support@create-llm.dev)
- 🐛 [GitHub Issues](https://github.com/yourusername/create-llm/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to create-llm! 🚀**
