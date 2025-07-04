#!/usr/bin/env python3
"""
GitHub Upload Helper for TRINETRA-Core
Handles common Git issues and prepares the project for GitHub upload
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil

def run_command(command, check=True, capture_output=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=capture_output, 
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_git_installation():
    """Check if Git is installed"""
    success, stdout, stderr = run_command("git --version")
    if success:
        print(f"✅ Git is installed: {stdout.strip()}")
        return True
    else:
        print("❌ Git is not installed. Please install Git first.")
        return False

def setup_git_config():
    """Setup Git configuration if not already set"""
    print("\n🔧 Setting up Git configuration...")
    
    # Check if user.name is set
    success, stdout, stderr = run_command("git config --global user.name")
    if not success or not stdout.strip():
        name = input("Enter your GitHub username: ")
        run_command(f'git config --global user.name "{name}"')
        print(f"✅ Set Git username to: {name}")
    else:
        print(f"✅ Git username already set: {stdout.strip()}")
    
    # Check if user.email is set
    success, stdout, stderr = run_command("git config --global user.email")
    if not success or not stdout.strip():
        email = input("Enter your GitHub email: ")
        run_command(f'git config --global user.email "{email}"')
        print(f"✅ Set Git email to: {email}")
    else:
        print(f"✅ Git email already set: {stdout.strip()}")

def clean_project():
    """Clean up project files that shouldn't be uploaded"""
    print("\n🧹 Cleaning up project files...")
    
    # Remove common problematic files/folders
    items_to_remove = [
        '__pycache__',
        '*.pyc',
        '.pytest_cache',
        'node_modules',
        '.DS_Store',
        'Thumbs.db',
        '*.log',
        '.env.local',
        '.env.production'
    ]
    
    removed_count = 0
    for root, dirs, files in os.walk('.'):
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path, ignore_errors=True)
            removed_count += 1
            print(f"🗑️  Removed: {pycache_path}")
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                removed_count += 1
                print(f"🗑️  Removed: {file_path}")
    
    if removed_count > 0:
        print(f"✅ Cleaned up {removed_count} files/directories")
    else:
        print("✅ No files needed cleaning")

def check_file_sizes():
    """Check for large files that might cause issues"""
    print("\n📏 Checking for large files...")
    
    large_files = []
    max_size = 100 * 1024 * 1024  # 100MB limit
    
    for root, dirs, files in os.walk('.'):
        # Skip venv directory
        if 'venv' in root or '.git' in root:
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > max_size:
                    large_files.append((file_path, size))
            except OSError:
                continue
    
    if large_files:
        print("⚠️  Found large files that might cause issues:")
        for file_path, size in large_files:
            size_mb = size / (1024 * 1024)
            print(f"   📁 {file_path}: {size_mb:.1f}MB")
        print("\n💡 Consider adding these to .gitignore or using Git LFS")
    else:
        print("✅ No large files found")

def init_git_repo():
    """Initialize Git repository"""
    print("\n📁 Initializing Git repository...")
    
    if os.path.exists('.git'):
        print("✅ Git repository already exists")
        return True
    
    success, stdout, stderr = run_command("git init")
    if success:
        print("✅ Git repository initialized")
        return True
    else:
        print(f"❌ Failed to initialize Git repository: {stderr}")
        return False

def create_readme():
    """Create or update README.md"""
    print("\n📝 Checking README.md...")
    
    readme_path = Path("README.md")
    if readme_path.exists():
        print("✅ README.md already exists")
        return True
    
    readme_content = """# TRINETRA-Core

A streamlined smart retail surveillance system focused on customer tracking, identification, and behavioral analytics.

## Features

- 🎯 **Enhanced Face Recognition**: Advanced face detection and recognition with streaming dataset support
- 📊 **Entrance Tracking**: Multi-camera people counting and journey tracking
- 🧠 **Behavioral Analytics**: Real-time pattern analysis and customer insights
- 🌐 **Modern Web Interface**: Streamlit-based dashboard with interactive visualizations
- 🔌 **REST API**: FastAPI backend for integration with other systems

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/TRINETRA-Core.git
   cd TRINETRA-Core
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python quickstart.py
   ```

## Project Structure

```
TRINETRA-Core/
├── src/
│   ├── backend/          # FastAPI backend
│   ├── frontend/         # Streamlit frontend
│   └── core_modules/     # Core AI modules
├── datasets/             # Sample datasets
├── tests/               # Test files
└── docs/                # Documentation
```

## Usage

- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print("✅ Created README.md")
    return True

def add_files_to_git():
    """Add files to Git staging area"""
    print("\n📦 Adding files to Git...")
    
    # Add all files except those in .gitignore
    success, stdout, stderr = run_command("git add .")
    if success:
        print("✅ Files added to Git staging area")
        
        # Show what files were added
        success, stdout, stderr = run_command("git status --porcelain")
        if success and stdout.strip():
            print("\n📄 Files to be committed:")
            for line in stdout.strip().split('\n'):
                if line.startswith('A'):
                    print(f"   ✅ {line[3:]}")
                elif line.startswith('M'):
                    print(f"   📝 {line[3:]}")
        
        return True
    else:
        print(f"❌ Failed to add files: {stderr}")
        return False

def commit_changes():
    """Commit changes to Git"""
    print("\n💾 Committing changes...")
    
    # Check if there are any changes to commit
    success, stdout, stderr = run_command("git diff --cached --name-only")
    if not success or not stdout.strip():
        print("⚠️  No changes to commit")
        return True
    
    commit_message = "Initial commit: TRINETRA-Core streamlined surveillance system"
    success, stdout, stderr = run_command(f'git commit -m "{commit_message}"')
    
    if success:
        print(f"✅ Changes committed: {commit_message}")
        return True
    else:
        print(f"❌ Failed to commit changes: {stderr}")
        return False

def setup_github_remote():
    """Setup GitHub remote repository"""
    print("\n🌐 Setting up GitHub remote...")
    
    # Check if remote already exists
    success, stdout, stderr = run_command("git remote -v")
    if success and "origin" in stdout:
        print("✅ Remote 'origin' already exists")
        print(f"Current remote: {stdout.strip()}")
        return True
    
    print("\n📝 To upload to GitHub, you need to:")
    print("1. Create a new repository on GitHub.com")
    print("2. Copy the repository URL")
    print("3. Enter it below")
    
    repo_url = input("\nEnter your GitHub repository URL (e.g., https://github.com/username/TRINETRA-Core.git): ")
    
    if not repo_url.strip():
        print("❌ No repository URL provided")
        return False
    
    success, stdout, stderr = run_command(f"git remote add origin {repo_url}")
    if success:
        print(f"✅ Remote 'origin' added: {repo_url}")
        return True
    else:
        print(f"❌ Failed to add remote: {stderr}")
        return False

def push_to_github():
    """Push changes to GitHub"""
    print("\n🚀 Pushing to GitHub...")
    
    # First, try to push to main branch
    success, stdout, stderr = run_command("git push -u origin main")
    if success:
        print("✅ Successfully pushed to GitHub (main branch)")
        return True
    
    # If main branch doesn't work, try master
    success, stdout, stderr = run_command("git push -u origin master")
    if success:
        print("✅ Successfully pushed to GitHub (master branch)")
        return True
    
    # If neither works, try creating and pushing main branch
    run_command("git branch -M main")
    success, stdout, stderr = run_command("git push -u origin main")
    if success:
        print("✅ Successfully pushed to GitHub (created main branch)")
        return True
    
    print(f"❌ Failed to push to GitHub: {stderr}")
    print("\n💡 Common solutions:")
    print("1. Make sure you have created the repository on GitHub")
    print("2. Check your GitHub credentials")
    print("3. Make sure the repository URL is correct")
    print("4. Try: git push -u origin main --force (if you're sure)")
    
    return False

def main():
    """Main function to handle GitHub upload"""
    print("🚀 TRINETRA-Core GitHub Upload Helper")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Step 1: Check Git installation
    if not check_git_installation():
        return False
    
    # Step 2: Setup Git configuration
    setup_git_config()
    
    # Step 3: Clean up project
    clean_project()
    
    # Step 4: Check for large files
    check_file_sizes()
    
    # Step 5: Initialize Git repository
    if not init_git_repo():
        return False
    
    # Step 6: Create README if needed
    if not create_readme():
        return False
    
    # Step 7: Add files to Git
    if not add_files_to_git():
        return False
    
    # Step 8: Commit changes
    if not commit_changes():
        return False
    
    # Step 9: Setup GitHub remote
    if not setup_github_remote():
        return False
    
    # Step 10: Push to GitHub
    if not push_to_github():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Successfully uploaded to GitHub!")
    print("🌐 Your repository should now be available on GitHub")
    print("📋 Next steps:")
    print("1. Visit your GitHub repository")
    print("2. Add a description and topics")
    print("3. Consider adding a license")
    print("4. Set up GitHub Actions for CI/CD")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
