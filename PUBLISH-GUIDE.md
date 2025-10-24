# 🚀 Publishing Guide

## Pre-Publish Checklist

### 1. Update package.json

```json
{
  "name": "create-llm",
  "version": "1.0.0",
  "author": "Your Name <your.email@example.com>",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourusername/create-llm.git"
  },
  "bugs": {
    "url": "https://github.com/yourusername/create-llm/issues"
  },
  "homepage": "https://github.com/yourusername/create-llm#readme"
}
```

**Replace:**
- `Your Name` with your actual name
- `your.email@example.com` with your email
- `yourusername` with your GitHub username

### 2. Final Tests

```bash
# Build
npm run build

# Test all templates
node dist/index.js test-nano --template nano --skip-install
node dist/index.js test-tiny --template tiny --skip-install
node dist/index.js test-small --template small --skip-install

# Test interactive mode
node dist/index.js
```

### 3. Verify Files

```bash
# Check what will be published
npm pack --dry-run

# Should include:
# - dist/
# - templates/
# - README.md
# - LICENSE
# - package.json
```

---

## Publishing to npm

### Step 1: Create npm Account

If you don't have one:
1. Go to https://www.npmjs.com/signup
2. Create account
3. Verify email

### Step 2: Login

```bash
npm login
```

Enter:
- Username
- Password
- Email
- 2FA code (if enabled)

### Step 3: Check Package Name

```bash
# Check if name is available
npm search create-llm
```

If taken, update `package.json`:
```json
{
  "name": "@yourusername/create-llm"
}
```

### Step 4: Publish

```bash
# Dry run first
npm publish --dry-run

# Publish for real
npm publish

# Or with scoped package
npm publish --access public
```

### Step 5: Verify

```bash
# Test installation
npx create-llm@latest test-verify --template nano
```

---

## Post-Publish

### 1. Create GitHub Release

```bash
# Tag the release
git tag v1.0.0
git push origin v1.0.0

# Or create annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

Then on GitHub:
1. Go to Releases
2. Click "Create a new release"
3. Select tag `v1.0.0`
4. Title: "v1.0.0 - Initial Release"
5. Copy content from CHANGELOG.md
6. Publish release

### 2. Update README Badges

Add to top of README.md:

```markdown
[![npm version](https://img.shields.io/npm/v/create-llm.svg)](https://www.npmjs.com/package/create-llm)
[![npm downloads](https://img.shields.io/npm/dm/create-llm.svg)](https://www.npmjs.com/package/create-llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/create-llm.svg)](https://github.com/yourusername/create-llm/stargazers)
```

### 3. Announce

#### Twitter/X
```
🚀 Just launched create-llm v1.0!

The fastest way to start training your own Language Model.

✨ 4 templates (1M to 1B params)
🔧 Complete training infrastructure
📊 Live dashboard
💬 Interactive chat
🎨 Plugin system

Try it: npx create-llm

#MachineLearning #AI #LLM #OpenSource
```

#### Reddit

Post to:
- r/MachineLearning
- r/learnmachinelearning
- r/Python
- r/programming

Title: "I built create-llm - A CLI tool to scaffold LLM training projects in seconds"

#### Dev.to / Medium

Write a blog post:
- Why you built it
- How it works
- Example walkthrough
- Future plans

#### Product Hunt

Submit to Product Hunt:
- Catchy tagline
- Screenshots
- Demo video
- Clear description

### 4. Set Up Monitoring

#### npm Stats
- Check https://npm-stat.com/charts.html?package=create-llm

#### GitHub Insights
- Watch stars, forks, issues
- Respond to issues promptly

#### User Feedback
- Monitor npm reviews
- Check GitHub discussions
- Respond to tweets/posts

---

## Maintenance

### Regular Updates

```bash
# Bug fix
npm version patch  # 1.0.0 -> 1.0.1

# New feature
npm version minor  # 1.0.0 -> 1.1.0

# Breaking change
npm version major  # 1.0.0 -> 2.0.0

# Publish update
npm publish
```

### Update CHANGELOG.md

```markdown
## [1.0.1] - 2025-01-25

### Fixed
- Bug fix description

### Added
- New feature description
```

### Respond to Issues

- Triage within 24 hours
- Fix critical bugs ASAP
- Label appropriately
- Be friendly and helpful

---

## Troubleshooting

### "Package name already exists"

**Solution:** Use scoped package
```json
{
  "name": "@yourusername/create-llm"
}
```

### "You must be logged in"

**Solution:**
```bash
npm logout
npm login
```

### "403 Forbidden"

**Solution:** Check package name availability
```bash
npm search create-llm
```

### "ENEEDAUTH"

**Solution:** Enable 2FA and use auth token
```bash
npm login --auth-type=web
```

---

## Success Checklist

- [ ] package.json updated with your info
- [ ] All tests passing
- [ ] README.md complete
- [ ] LICENSE file present
- [ ] CHANGELOG.md updated
- [ ] Built successfully (`npm run build`)
- [ ] Tested locally
- [ ] npm account created
- [ ] Logged into npm
- [ ] Package name available
- [ ] Published to npm
- [ ] GitHub release created
- [ ] Badges added to README
- [ ] Announced on social media
- [ ] Monitoring set up

---

## 🎉 Congratulations!

Your package is now live on npm!

**Next Steps:**
1. Monitor for issues
2. Respond to feedback
3. Plan next features
4. Build community
5. Keep improving

**Resources:**
- [npm Documentation](https://docs.npmjs.com/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Open Source Guide](https://opensource.guide/)

---

**Good luck with your launch! 🚀**
