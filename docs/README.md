# create-llm Documentation

This directory contains the Mintlify documentation for create-llm.

## Setup

1. Install Mintlify CLI:
```bash
npm i -g mintlify
```

2. Preview documentation locally:
```bash
cd docs
mintlify dev
```

3. Open http://localhost:3000 in your browser

## Deployment

### Deploy to Mintlify

1. Sign up at [mintlify.com](https://mintlify.com)
2. Connect your GitHub repository
3. Mintlify will automatically deploy on push to main

### Custom Domain (Optional)

Add a custom domain in your Mintlify dashboard settings.

## Structure

```
docs/
├── mint.json              # Main configuration
├── introduction.mdx       # Home page
├── quickstart.mdx         # Getting started guide
├── installation.mdx       # Installation instructions
├── templates/             # Template documentation
│   ├── overview.mdx
│   ├── nano.mdx
│   ├── tiny.mdx
│   ├── small.mdx
│   └── base.mdx
├── concepts/              # Core concepts
├── guides/                # How-to guides
├── config/                # Configuration reference
├── api-reference/         # API documentation
├── examples/              # Example projects
└── troubleshooting/       # Common issues
```

## Adding New Pages

1. Create a new `.mdx` file in the appropriate directory
2. Add frontmatter:
```mdx
---
title: 'Page Title'
description: 'Page description'
---
```
3. Add the page to `mint.json` navigation
4. Commit and push

## Writing Tips

- Use MDX components: `<Card>`, `<CardGroup>`, `<Tabs>`, `<Steps>`, etc.
- Add code blocks with syntax highlighting
- Include images in `/images` directory
- Use callouts: `<Tip>`, `<Warning>`, `<Info>`, `<Note>`

## Resources

- [Mintlify Documentation](https://mintlify.com/docs)
- [MDX Components](https://mintlify.com/docs/content/components)
- [Configuration](https://mintlify.com/docs/settings/global)
