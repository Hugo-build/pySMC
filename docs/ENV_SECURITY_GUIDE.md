# Environment Variable Security Guide

## 🔐 Encrypting Your .env File

You have several options to secure your API keys:

## Option 1: Simple Encryption Script (Recommended for Solo Projects)

### Setup:
```bash
# Install dependencies
uv sync  # or pip install cryptography

# Make the script executable
chmod +x encrypt_env.py
```

### Usage:

**Encrypt your .env file:**
```bash
python encrypt_env.py encrypt
# Enter a strong password when prompted
# This creates: .env.encrypted and .env.salt
```

**Delete the original .env:**
```bash
rm .env  # Only do this after confirming encryption worked!
```

**Decrypt when needed:**
```bash
python encrypt_env.py decrypt
# Enter your password
# This recreates .env from .env.encrypted
```

**Run your app:**
```bash
streamlit run app.py
```

### What Gets Committed to Git:
- ✅ `.env.encrypted` - Your encrypted secrets (safe to commit)
- ✅ `.env.salt` - Salt for encryption (safe to commit)
- ❌ `.env` - Original file (in .gitignore, never commit!)

---

## Option 2: System Keychain (macOS/Linux)

Use your OS keychain to store secrets:

```python
# Install: pip install keyring
import keyring

# Store API key (one time)
keyring.set_password("pysmc", "openai_api_key", "your-api-key-here")

# Retrieve in app.py
import keyring
OPENAI_API_KEY = keyring.get_password("pysmc", "openai_api_key")
```

Update `app.py`:
```python
import keyring
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or keyring.get_password("pysmc", "openai_api_key")
```

---

## Option 3: Git-Crypt (For Team Projects)

Automatically encrypts files in git:

```bash
# Install git-crypt
brew install git-crypt  # macOS
# or: apt-get install git-crypt  # Linux

# Initialize in your repo
cd /path/to/pySMC
git-crypt init

# Configure which files to encrypt
echo ".env filter=git-crypt diff=git-crypt" >> .gitattributes
echo ".env.* filter=git-crypt diff=git-crypt" >> .gitattributes

# Add collaborators
git-crypt add-gpg-user USER_GPG_ID

# Files are automatically encrypted when committed!
```

---

## Option 4: Cloud Secret Manager (Production)

For production deployments:

### Google Secret Manager:
```python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
name = "projects/PROJECT_ID/secrets/openai-api-key/versions/latest"
response = client.access_secret_version(request={"name": name})
OPENAI_API_KEY = response.payload.data.decode("UTF-8")
```

### AWS Secrets Manager:
```python
import boto3

client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='openai-api-key')
OPENAI_API_KEY = response['SecretString']
```

---

## Option 5: Environment-Only (No File)

Don't use `.env` file at all:

```bash
# Set environment variables directly
export OPENAI_API_KEY="your-key-here"
export OPENAI_MODEL="gemini-2.0-flash-exp"
export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

# Add to ~/.zshrc or ~/.bashrc for persistence
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.zshrc
```

---

## 🛡️ Security Best Practices

### DO:
- ✅ Use encrypted .env files for version control
- ✅ Rotate API keys regularly
- ✅ Use different keys for dev/staging/production
- ✅ Keep .env in .gitignore
- ✅ Use environment variables in deployment
- ✅ Share .env.example (without actual keys)

### DON'T:
- ❌ Never commit .env with real keys
- ❌ Don't share API keys in chat/email
- ❌ Don't hardcode keys in source files
- ❌ Don't use production keys for development
- ❌ Don't share encryption passwords over insecure channels

---

## 🔄 Workflow for Team Collaboration

### For Team Members:

**Person A (has API key):**
```bash
# 1. Add API key to .env
# 2. Encrypt it
python encrypt_env.py encrypt
# 3. Commit encrypted files
git add .env.encrypted .env.salt
git commit -m "Add encrypted environment variables"
git push
# 4. Share password securely (1Password, LastPass, etc.)
```

**Person B (new team member):**
```bash
# 1. Clone repo
git clone <repo-url>
cd pySMC
# 2. Install dependencies
uv sync
# 3. Get password from team (secure channel)
# 4. Decrypt .env
python encrypt_env.py decrypt
# 5. Run app
streamlit run app.py
```

---

## 🧪 Testing Your Encryption

```bash
# 1. Encrypt
python encrypt_env.py encrypt

# 2. Backup and delete original
cp .env .env.backup
rm .env

# 3. Try to run app (should fail - no .env)
streamlit run app.py  # ❌ Will error

# 4. Decrypt
python encrypt_env.py decrypt

# 5. Run app (should work)
streamlit run app.py  # ✅ Works!

# 6. Compare files
diff .env .env.backup  # Should be identical
```

---

## 🆘 Troubleshooting

**Forgot encryption password?**
- If you have the original .env backed up: restore it
- Otherwise: regenerate API keys and create new .env

**Wrong password error?**
- Check for typos
- Ensure you're using the same password used to encrypt
- Verify .env.salt file exists

**Can't run app after decryption?**
- Check .env file was created: `ls -la .env`
- Verify contents: `cat .env` (be careful not to share output!)
- Check file permissions: `chmod 600 .env`

---

## 📋 Quick Reference

| Command | Purpose |
|---------|---------|
| `python encrypt_env.py encrypt` | Encrypt .env → .env.encrypted |
| `python encrypt_env.py decrypt` | Decrypt .env.encrypted → .env |
| `git add .env.encrypted .env.salt` | Commit encrypted files |
| `rm .env` | Remove plaintext (after encryption) |

**Remember:** Keep your encryption password safe! Store it in a password manager.


