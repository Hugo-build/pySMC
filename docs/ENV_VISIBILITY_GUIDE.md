# .env File Visibility & Protection Guide

## 🔒 How .env Files Are Protected

Your `.env` file has **multiple layers of protection** to keep your API keys safe:

---

## Layer 1: Filesystem Hidden (Automatic)

### Why it's named `.env` (with a dot)

Files starting with `.` are **hidden by default** on Unix/Linux/macOS:

```bash
# Normal ls command - .env is HIDDEN
$ ls
app.py  server_demo.py  test_data.json

# Need -a flag to see hidden files
$ ls -a
.env  .env.example  .gitignore  app.py  server_demo.py  test_data.json

# Or use -A (shows hidden except . and ..)
$ ls -A
```

### File Manager Behavior

**macOS Finder:**
- `.env` is invisible by default
- Press `Cmd + Shift + .` to toggle visibility
- Hidden files appear dimmed

**VS Code / Cursor:**
- Shows hidden files by default (for developers)
- You NEED to see `.env` to edit it!

---

## Layer 2: Git Ignore (Protected from Commits)

Your `.gitignore` file prevents `.env` from being committed:

```bash
# Check if .env is ignored
$ git status --ignored | grep .env
.env        # ✅ Shows as ignored - won't be committed
```

### Test Protection:

```bash
# Try to add .env (will be ignored)
$ git add .env
# Git will silently ignore it ✅

# Check what would be committed
$ git status
# .env won't appear in the list ✅

# If .env appears, it means it's NOT in .gitignore (BAD!)
```

### Your Current .gitignore:

```
# environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
.env.development
.env.test
.env.production
```

✅ All these patterns are protected!

---

## Layer 3: File Permissions (Unix/Linux/macOS)

Make `.env` readable only by you:

```bash
# Set strict permissions (owner read/write only)
$ chmod 600 .env

# Verify
$ ls -la .env
-rw-------  1 yuma  staff  991 Nov  5 11:36 .env
#  ^^^ Only owner can read/write
```

Permission breakdown:
- `600` = Owner: read+write, Group: none, Others: none
- `-rw-------` = Only you can access it

---

## Layer 4: IDE/Editor Protection

### VS Code / Cursor Settings:

Add to `.vscode/settings.json` or Cursor settings:

```json
{
  "files.exclude": {
    "**/.env": false  // Show .env (you need to edit it)
  },
  "files.watcherExclude": {
    "**/.env": true  // Don't watch for changes (performance)
  },
  "search.exclude": {
    "**/.env": true  // Exclude from search results
  }
}
```

This prevents accidentally searching and exposing keys.

---

## 🧪 Test Your Protection

Run these tests to verify `.env` is properly hidden:

### Test 1: Filesystem Hidden
```bash
cd /path/to/pySMC
ls | grep .env
# Should return NOTHING (file is hidden) ✅
```

### Test 2: Git Ignoring
```bash
git add .env
git status
# .env should NOT appear in "Changes to be committed" ✅
```

### Test 3: Not in Remote Repo
```bash
git log --all --full-history -- .env
# Should return NOTHING (never committed) ✅
```

### Test 4: Check Remote (GitHub/GitLab)
```bash
# Visit your repo on GitHub
# Search for ".env" in the file tree
# Should NOT find it ✅
```

---

## ⚠️ Common Mistakes & Fixes

### Mistake 1: .env Was Already Committed

If you accidentally committed `.env` before:

```bash
# Remove from git history (but keep local file)
git rm --cached .env

# Commit the removal
git commit -m "Remove .env from version control"

# Add to .gitignore (if not already there)
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to .gitignore"

# Push changes
git push
```

**Warning:** The old commits still have the keys! Rotate your API keys immediately.

### Mistake 2: .env is Visible in Git Status

```bash
$ git status
# Shows: .env (modified)   # ❌ BAD!

# Fix: Add to .gitignore
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Ignore .env file"
```

### Mistake 3: Can't Find .env File

```bash
# Show hidden files
ls -a | grep .env

# If not found, check if it was encrypted
ls -a | grep "env"
# Look for: .env.encrypted, .env.salt

# Decrypt if needed
python encrypt_env.py decrypt
```

---

## 🛡️ Best Practices Summary

### DO:
✅ Use `.env` (with dot) for automatic hiding
✅ Keep `.env` in `.gitignore`
✅ Set file permissions: `chmod 600 .env`
✅ Share `.env.example` (template without keys)
✅ Use encryption for backups: `python encrypt_env.py encrypt`
✅ Rotate keys if accidentally exposed

### DON'T:
❌ Never use `env` (without dot) - not hidden!
❌ Never commit `.env` to git
❌ Never share `.env` via email/chat
❌ Never screenshot `.env` contents
❌ Never include API keys in code
❌ Never use production keys for development

---

## 📋 Quick Reference

| Command | Purpose |
|---------|---------|
| `ls -a` | Show hidden files (including .env) |
| `git status --ignored` | Check if .env is ignored |
| `chmod 600 .env` | Set strict permissions |
| `git rm --cached .env` | Remove from git (keep local) |
| `python encrypt_env.py encrypt` | Encrypt .env for backup |

---

## 🔍 How to View .env Safely

### In Terminal:
```bash
# View file (don't share output!)
cat .env

# Edit file
nano .env
# or
code .env  # VS Code
```

### In Finder (macOS):
```bash
# Toggle hidden files visibility
Cmd + Shift + .

# Or open directly
open -a TextEdit .env
```

### In Your IDE:
- VS Code/Cursor shows hidden files by default
- Just open `.env` in the editor
- Files starting with `.` may appear dimmed

---

## 🚨 Emergency: API Key Exposed

If you accidentally exposed your API key:

1. **Immediately rotate the key:**
   - OpenAI: https://platform.openai.com/api-keys
   - Google: https://aistudio.google.com/apikey

2. **Check git history:**
   ```bash
   git log --all --full-history -- .env
   ```

3. **Remove from git if found:**
   ```bash
   git rm --cached .env
   git commit -m "Remove exposed credentials"
   ```

4. **Update .env with new key**

5. **Consider using git-crypt or encryption going forward**

---

## ✅ Verify Your Setup

Run this checklist:

```bash
# 1. File is hidden
ls | grep -q .env && echo "❌ VISIBLE" || echo "✅ Hidden"

# 2. File is ignored by git
git check-ignore .env && echo "✅ Ignored" || echo "❌ NOT IGNORED"

# 3. Has strict permissions
[[ $(stat -f "%A" .env 2>/dev/null || stat -c "%a" .env 2>/dev/null) == "600" ]] && echo "✅ Secure" || echo "⚠️ Consider: chmod 600 .env"

# 4. Not in git history
[[ -z $(git log --all --full-history -- .env) ]] && echo "✅ Never committed" || echo "❌ WAS COMMITTED - ROTATE KEYS!"
```

All checks should pass ✅ for maximum security!


