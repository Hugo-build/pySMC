# 🔐 Two-Factor Authentication (2FA) Setup Guide

Protect your `.env` file with **password + 2FA** (Google Authenticator style).

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
uv sync
# This installs: pyotp, qrcode, cryptography
```

### Step 2: Initial 2FA Setup (One Time)

```bash
python encrypt_env_2fa.py setup
```

This will:
1. Generate a unique 2FA secret for your project
2. Show a QR code in your terminal
3. Prompt you to scan it with your authenticator app
4. Ask for a master password
5. Save encrypted 2FA secret to `.env.2fa.secret`

**📱 Recommended Authenticator Apps:**
- Google Authenticator (iOS/Android)
- Authy (iOS/Android/Desktop)
- 1Password (has TOTP built-in)
- Microsoft Authenticator
- Any TOTP-compatible app

### Step 3: Encrypt Your .env

```bash
python encrypt_env_2fa.py encrypt
# Enter password + 6-digit code from your app
```

This creates:
- `.env.encrypted` - Your encrypted secrets
- `.env.salt` - Encryption salt
- `.env.2fa.secret` - Encrypted TOTP secret (already created in setup)

### Step 4: Delete Original (Optional)

```bash
# Backup first!
cp .env .env.backup

# Then delete
rm .env
```

### Step 5: Decrypt When Needed

```bash
python encrypt_env_2fa.py decrypt
# Enter password + current 6-digit code
```

This recreates `.env` so your app can use it.

---

## 📋 Complete Workflow

### First Time Setup:

```bash
# 1. Create .env with your API keys
echo "OPENAI_API_KEY=your-key-here" > .env

# 2. Install dependencies
uv sync

# 3. Set up 2FA (scan QR code)
python encrypt_env_2fa.py setup

# 4. Encrypt the .env file
python encrypt_env_2fa.py encrypt

# 5. (Optional) Remove plain .env
rm .env

# 6. Commit encrypted files
git add .env.encrypted .env.salt .env.2fa.secret
git commit -m "Add encrypted environment with 2FA"
```

### Daily Usage:

```bash
# Morning: Decrypt .env
python encrypt_env_2fa.py decrypt
# Enter: password + 6-digit code

# Work: Use your app
streamlit run app.py

# Evening: Re-encrypt (optional)
python encrypt_env_2fa.py encrypt
rm .env
```

### Team Member Onboarding:

```bash
# 1. Clone repo
git clone <repo>
cd pySMC

# 2. Install dependencies
uv sync

# 3. Get password & 2FA secret from team securely
# (They should send you the 2FA QR code or secret key)

# 4. Set up YOUR 2FA app with the shared secret

# 5. Decrypt
python encrypt_env_2fa.py decrypt
# Enter: password + your 6-digit code

# 6. Run app
streamlit run app.py
```

---

## 🔒 How It Works

### Three Layers of Security:

1. **Password** - Something you know
2. **2FA Code** - Something you have (your phone/authenticator)
3. **Encrypted Files** - Safe to commit to git

### Encryption Flow:

```
.env (plain text)
    ↓
[Password + 2FA Code] 
    ↓
Encryption Key = PBKDF2(password + TOTP_secret + code)
    ↓
.env.encrypted (AES-256)
```

### What Gets Stored:

| File | Contents | Can Commit? |
|------|----------|------------|
| `.env` | Plain API keys | ❌ NO (gitignored) |
| `.env.encrypted` | Encrypted API keys | ✅ YES |
| `.env.salt` | Random salt | ✅ YES |
| `.env.2fa.secret` | Encrypted TOTP secret | ✅ YES |

---

## 🧪 Testing Your 2FA

### Verify Your Code Works:

```bash
python encrypt_env_2fa.py verify
# Shows current valid code
# Tests if your app is synced
```

### Manual Test:

```bash
# 1. Get current code from your app
# Example: 123456

# 2. Try to decrypt
python encrypt_env_2fa.py decrypt
# If it works → 2FA is set up correctly ✅
```

---

## 🔧 Commands Reference

| Command | What It Does |
|---------|-------------|
| `python encrypt_env_2fa.py setup` | Initial 2FA setup (QR code) |
| `python encrypt_env_2fa.py encrypt` | Encrypt .env → .env.encrypted |
| `python encrypt_env_2fa.py decrypt` | Decrypt .env.encrypted → .env |
| `python encrypt_env_2fa.py verify` | Test your 2FA code |

---

## 🆘 Troubleshooting

### "Invalid 2FA code"

**Causes:**
- Code expired (they change every 30 seconds)
- Phone clock is wrong (TOTP is time-based)
- Wrong authenticator account selected

**Solutions:**
```bash
# Check if your code is valid
python encrypt_env_2fa.py verify

# Sync your phone's clock
# Settings → Date & Time → Automatic

# Re-scan QR code
python encrypt_env_2fa.py setup  # Choose "yes" to reset
```

### "Invalid password"

- Check for typos
- Try again
- If forgotten: You'll need to regenerate API keys

### "2FA not set up"

```bash
# Run setup first
python encrypt_env_2fa.py setup
```

### Lost Phone / New Device

**If you have backups:**
```bash
# Restore from backup
cp .env.backup .env
python encrypt_env_2fa.py setup  # New 2FA
python encrypt_env_2fa.py encrypt
```

**If you don't:**
- Regenerate API keys from providers
- Create new .env
- Run setup again

---

## 🛡️ Security Best Practices

### DO:
✅ Use a strong master password (12+ characters)
✅ Keep 2FA secret in authenticator app
✅ Backup your 2FA secret (encrypted form is safe)
✅ Use different 2FA for different projects
✅ Commit `.env.encrypted`, `.env.salt`, `.env.2fa.secret`
✅ Share 2FA setup with team via secure channel

### DON'T:
❌ Don't share master password in plain text
❌ Don't screenshot/share 2FA QR codes publicly
❌ Don't commit `.env` (plain text)
❌ Don't use same password for multiple projects
❌ Don't disable 2FA after setting it up
❌ Don't lose your phone without backup

---

## 🔄 Migration from Simple Encryption

Already using `encrypt_env.py`? Upgrade to 2FA:

```bash
# 1. Decrypt with old method
python encrypt_env.py decrypt

# 2. Set up 2FA
python encrypt_env_2fa.py setup

# 3. Encrypt with 2FA
python encrypt_env_2fa.py encrypt

# 4. Clean up old files
rm .env.encrypted.old .env.salt.old

# 5. Use new script going forward
python encrypt_env_2fa.py decrypt
```

---

## 📱 Sharing 2FA with Team

### Option 1: Share QR Code (Secure Channel)

```bash
# During setup, take a screenshot of the QR code
# Send via encrypted channel (1Password, Signal, etc.)
# Team members scan the SAME QR code
```

### Option 2: Share Secret Key

```bash
# During setup, copy the secret key shown
# Example: JBSWY3DPEHPK3PXP

# Team member manually enters in their authenticator:
# - Account: pySMC@env
# - Secret: JBSWY3DPEHPK3PXP
# - Type: Time-based
```

### Option 3: Export from 1Password

```bash
# If using 1Password:
# 1. Save TOTP in 1Password
# 2. Share vault item with team
# 3. Everyone syncs the same TOTP
```

---

## 🎯 Which Should I Use?

| Scenario | Recommendation |
|----------|---------------|
| Solo developer, basic security | `encrypt_env.py` (password only) |
| Team project, shared secrets | `encrypt_env_2fa.py` (password + 2FA) |
| High security requirements | `encrypt_env_2fa.py` + hardware key |
| Production deployment | Cloud secret manager (AWS/GCP) |

---

## 🚨 Emergency: Lost Access

### If you lose password:
- Can't decrypt → Need to regenerate API keys
- No backdoor (by design)

### If you lose 2FA:
- Can't decrypt → Need to regenerate API keys
- Keep backup of `.env` separately

### Prevention:
```bash
# Always keep an encrypted backup
cp .env .env.backup
chmod 600 .env.backup

# Store backup password in password manager
# Store 2FA QR code in secure notes

# Or use multiple 2FA devices
# (scan same QR code on multiple phones)
```

---

## ✅ Verification Checklist

Before committing encrypted files:

- [ ] `.env` is in `.gitignore`
- [ ] 2FA setup completed successfully
- [ ] Can decrypt with password + code
- [ ] Backup of plain `.env` stored securely
- [ ] Master password saved in password manager
- [ ] 2FA added to authenticator app
- [ ] Tested full encrypt/decrypt cycle
- [ ] Team members can access with shared credentials

---

## 🎓 Advanced: Hardware 2FA (YubiKey)

Want even more security? Combine with hardware keys:

```bash
# Install: pip install fido2

# Modify encrypt_env_2fa.py to require:
# 1. Password (something you know)
# 2. TOTP (something you have - phone)
# 3. YubiKey (something you have - hardware)
```

This would require ALL THREE to decrypt! 🔒🔒🔒


