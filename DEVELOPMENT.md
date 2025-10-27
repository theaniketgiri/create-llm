# Development Notes

## Project History

This project started as a personal tool to quickly scaffold LLM training projects. I was frustrated with the boilerplate needed to get started with PyTorch training, so I built this to automate it.

## Development Process

### Initial Development (v1.0.0)
- Built core scaffolding engine
- Created 4 templates (NANO, TINY, SMALL, BASE)
- Implemented basic CLI with Inquirer
- Added tokenizer training support

### Iteration Phase (v1.x)
- Added plugin system (WandB, HuggingFace, SynthexAI)
- Improved error handling
- Added live training dashboard
- Enhanced documentation

### Bug Fix Release (v2.0.1)
- Fixed critical position embedding IndexError
- Added automatic sequence length validation
- Improved error messages with diagnostics
- Added comprehensive test coverage

## Technical Decisions

### Why TypeScript for CLI?
- Fast startup time
- Easy npm distribution
- Type safety catches errors early
- Industry standard for CLI tools

### Why Template Strings?
- Simple and maintainable
- Common pattern (used by create-react-app, etc.)
- Self-contained (no external template files)
- Fast (no file I/O)

Trade-off: Mixing Python in TypeScript strings isn't ideal, but it's pragmatic for this project size.

### Why Not Python for Everything?
- Slower CLI startup
- More complex distribution
- npm ecosystem is better for CLI tools

## Known Issues & Technical Debt

### Git History
- Early commits have poor messages (learning git workflow)
- Some commits are too large
- Will improve going forward with conventional commits

### Code Quality
- Some template files are large (>1000 lines)
- Could benefit from refactoring to separate template files
- Test coverage could be better

### Documentation
- Some docs are verbose (AI-assisted)
- Could be more concise
- Will iterate based on user feedback

## Future Improvements

### Short-term
- [ ] Clean up commit messages
- [ ] Improve test coverage
- [ ] Refactor large template files
- [ ] Add more examples

### Long-term
- [ ] Separate template files (.py.template)
- [ ] Add more model architectures
- [ ] Distributed training support
- [ ] Better plugin system

## Lessons Learned

1. **Ship fast, iterate**: Better to have a working tool with messy commits than perfect commits with no tool
2. **User feedback matters**: Real users found bugs I didn't anticipate
3. **Documentation is hard**: Finding the right balance between comprehensive and concise
4. **AI is a tool**: Like any tool, it's about how you use it

## Transparency

I use AI (Claude, ChatGPT, Copilot) as development tools, similar to how I use Stack Overflow or documentation. The key differences:

**What I write:**
- Core architecture decisions
- Business logic
- Bug fixes
- Test cases

**What AI helps with:**
- Boilerplate code
- Documentation structure
- Template generation
- Repetitive patterns

**What I always do:**
- Review all AI-generated code
- Understand what it does
- Test thoroughly
- Maintain and debug

## Contributing

If you're interested in contributing, check out CONTRIBUTING.md. I welcome:
- Bug reports
- Feature requests
- Code contributions
- Documentation improvements

---

**Last Updated**: 2025-10-26
**Maintainer**: Aniket Giri
