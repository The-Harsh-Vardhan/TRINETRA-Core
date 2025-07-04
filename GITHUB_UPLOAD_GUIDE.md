# TRINETRA-Core GitHub Upload Guide

## Step-by-Step Instructions

### Prerequisites

1. **Install Git** (if not already installed):

   - Download from: https://git-scm.com/downloads
   - Install with default options

2. **Create GitHub Account** (if not already have one):
   - Go to: https://github.com
   - Sign up for free

### Step 1: Clean Your Project

Open Command Prompt/PowerShell in your project directory and run:

```bash
python github_upload_helper.py
```

Or double-click on: `upload_to_github.bat`

### Step 2: Manual Git Setup (if the script doesn't work)

1. **Initialize Git repository**:

   ```bash
   git init
   ```

2. **Configure Git** (first time only):

   ```bash
   git config --global user.name "Your GitHub Username"
   git config --global user.email "your.email@example.com"
   ```

3. **Add files to staging**:

   ```bash
   git add .
   ```

4. **Commit changes**:
   ```bash
   git commit -m "Initial commit: TRINETRA-Core surveillance system"
   ```

### Step 3: Create GitHub Repository

1. Go to GitHub.com and log in
2. Click the "+" icon in the top right
3. Select "New repository"
4. Enter repository name: `TRINETRA-Core`
5. Add description: "Smart retail surveillance system with AI-powered analytics"
6. Choose "Public" or "Private"
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

### Step 4: Connect to GitHub

1. **Copy the repository URL** from GitHub (it will look like):

   ```
   https://github.com/yourusername/TRINETRA-Core.git
   ```

2. **Add remote origin**:

   ```bash
   git remote add origin https://github.com/yourusername/TRINETRA-Core.git
   ```

3. **Push to GitHub**:
   ```bash
   git branch -M main
   git push -u origin main
   ```

### Step 5: Verify Upload

1. Refresh your GitHub repository page
2. You should see all your files uploaded
3. The README.md should display your project description

## Common Issues and Solutions

### Issue 1: "Git is not recognized"

**Solution**: Install Git from https://git-scm.com/downloads

### Issue 2: "Permission denied" or "Authentication failed"

**Solution**:

- Use Personal Access Token instead of password
- Go to GitHub → Settings → Developer settings → Personal access tokens
- Generate new token with "repo" permissions
- Use token as password when prompted

### Issue 3: "Repository not found"

**Solution**:

- Make sure you created the repository on GitHub
- Check the repository URL is correct
- Ensure you have push permissions

### Issue 4: "Large files" error

**Solution**:

- Check what files are large: `git ls-files --others --ignored --exclude-standard`
- Add them to .gitignore
- Or use Git LFS for large files

### Issue 5: "Nothing to commit"

**Solution**:

- Check git status: `git status`
- Make sure files aren't in .gitignore
- Use `git add -f filename` to force add if needed

## Files Structure After Upload

Your GitHub repository should contain:

```
TRINETRA-Core/
├── .gitignore
├── README.md
├── requirements.txt
├── quickstart.py
├── package_manager.py
├── github_upload_helper.py
├── upload_to_github.bat
├── src/
│   ├── backend/
│   ├── frontend/
│   └── core_modules/
├── tests/
├── config/
└── github/
```

## Next Steps After Upload

1. **Add repository description** on GitHub
2. **Add topics/tags** (e.g., "python", "computer-vision", "surveillance", "ai")
3. **Star your own repository** to show it's active
4. **Consider adding a license** (MIT is recommended)
5. **Set up GitHub Actions** for CI/CD (optional)

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Google the specific error message
3. Check GitHub's documentation
4. Ask on Stack Overflow with tags [git] [github]

## Security Notes

- Never commit secrets, passwords, or API keys
- Use environment variables for sensitive data
- Review your .gitignore file regularly
- Consider using GitHub's secret scanning features
