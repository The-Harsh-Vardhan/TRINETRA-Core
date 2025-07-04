#!/usr/bin/env python3
"""
TRINETRA-Core GitHub Upload Troubleshooter
Identifies and helps fix common GitHub upload issues
"""

import os
import subprocess
import sys
from pathlib import Path

def run_cmd(command):
    """Run a command and return success, stdout, stderr"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_git():
    """Check if git is installed and working"""
    print("🔍 Checking Git installation...")
    
    success, stdout, stderr = run_cmd("git --version")
    if success:
        print(f"✅ Git is installed: {stdout.strip()}")
        return True
    else:
        print("❌ Git is not installed or not accessible")
        print("💡 Solution: Install Git from https://git-scm.com/downloads")
        return False

def check_git_config():
    """Check git configuration"""
    print("\n🔍 Checking Git configuration...")
    
    success, stdout, stderr = run_cmd("git config --global user.name")
    if success and stdout.strip():
        print(f"✅ Git username: {stdout.strip()}")
    else:
        print("❌ Git username not set")
        print("💡 Solution: git config --global user.name 'Your Name'")
    
    success, stdout, stderr = run_cmd("git config --global user.email")
    if success and stdout.strip():
        print(f"✅ Git email: {stdout.strip()}")
    else:
        print("❌ Git email not set")
        print("💡 Solution: git config --global user.email 'your.email@example.com'")

def check_git_repo():
    """Check if git repository is initialized"""
    print("\n🔍 Checking Git repository...")
    
    if os.path.exists('.git'):
        print("✅ Git repository exists")
        
        # Check if there are any commits
        success, stdout, stderr = run_cmd("git log --oneline -1")
        if success:
            print(f"✅ Repository has commits: {stdout.strip()}")
        else:
            print("⚠️  Repository has no commits yet")
            print("💡 Solution: git add . && git commit -m 'Initial commit'")
        
        # Check for remotes
        success, stdout, stderr = run_cmd("git remote -v")
        if success and stdout.strip():
            print(f"✅ Remote repositories configured:")
            for line in stdout.strip().split('\n'):
                print(f"   {line}")
        else:
            print("⚠️  No remote repositories configured")
            print("💡 Solution: git remote add origin https://github.com/username/repo.git")
        
        return True
    else:
        print("❌ Git repository not initialized")
        print("💡 Solution: git init")
        return False

def check_file_sizes():
    """Check for files that might be too large for GitHub"""
    print("\n🔍 Checking file sizes...")
    
    large_files = []
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        skip_dirs = ['.git', 'venv', '__pycache__', '.pytest_cache', 'node_modules']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
                
                # GitHub has a 100MB file limit
                if size > 100 * 1024 * 1024:  # 100MB
                    large_files.append((file_path, size))
                elif size > 50 * 1024 * 1024:  # 50MB (warning)
                    print(f"⚠️  Large file: {file_path} ({size/1024/1024:.1f}MB)")
                
            except (OSError, IOError):
                continue
    
    print(f"✅ Total files: {file_count}")
    print(f"✅ Total size: {total_size/1024/1024:.1f}MB")
    
    if large_files:
        print(f"❌ Found {len(large_files)} files over 100MB limit:")
        for file_path, size in large_files:
            print(f"   📁 {file_path}: {size/1024/1024:.1f}MB")
        print("💡 Solution: Add these files to .gitignore or use Git LFS")
        return False
    else:
        print("✅ No files exceed GitHub's size limits")
        return True

def check_gitignore():
    """Check .gitignore file"""
    print("\n🔍 Checking .gitignore...")
    
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        print("✅ .gitignore file exists")
        
        # Check for common patterns
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        important_patterns = [
            'venv/', '__pycache__/', '*.pyc', '.env', '*.log'
        ]
        
        missing_patterns = []
        for pattern in important_patterns:
            if pattern not in content:
                missing_patterns.append(pattern)
        
        if missing_patterns:
            print(f"⚠️  Missing important patterns: {', '.join(missing_patterns)}")
            print("💡 Consider adding these to .gitignore")
        else:
            print("✅ .gitignore has important patterns")
    else:
        print("⚠️  No .gitignore file found")
        print("💡 Solution: Create a .gitignore file")

def check_network():
    """Check if we can reach GitHub"""
    print("\n🔍 Checking network connectivity to GitHub...")
    
    try:
        import urllib.request
        import urllib.error
        
        # Try to reach GitHub
        urllib.request.urlopen('https://github.com', timeout=10)
        print("✅ Can reach GitHub")
        
        # Try to reach GitHub API
        urllib.request.urlopen('https://api.github.com', timeout=10)
        print("✅ Can reach GitHub API")
        
        return True
    except urllib.error.URLError as e:
        print(f"❌ Cannot reach GitHub: {e}")
        print("💡 Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Network check failed: {e}")
        return False

def check_credentials():
    """Check if credentials are working"""
    print("\n🔍 Checking GitHub credentials...")
    
    # This is a basic check - actual auth happens during push
    success, stdout, stderr = run_cmd("git config --global user.name")
    name_set = success and stdout.strip()
    
    success, stdout, stderr = run_cmd("git config --global user.email")
    email_set = success and stdout.strip()
    
    if name_set and email_set:
        print("✅ Basic credentials configured")
        print("💡 Note: GitHub authentication uses tokens, not passwords")
        print("💡 If push fails, you may need a Personal Access Token")
    else:
        print("❌ Basic credentials not configured")
        print("💡 Solution: Set up user.name and user.email")

def suggest_next_steps():
    """Suggest next steps based on the checks"""
    print("\n📋 Suggested Next Steps:")
    print("1. Fix any issues identified above")
    print("2. Create a new repository on GitHub.com")
    print("3. Run: git remote add origin https://github.com/username/repo.git")
    print("4. Run: git push -u origin main")
    print("\n📚 For detailed help, see: GITHUB_UPLOAD_GUIDE.md")

def main():
    """Main troubleshooting function"""
    print("🔧 TRINETRA-Core GitHub Upload Troubleshooter")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    all_good = True
    
    # Run all checks
    all_good &= check_git()
    check_git_config()
    all_good &= check_git_repo()
    all_good &= check_file_sizes()
    check_gitignore()
    all_good &= check_network()
    check_credentials()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! Your project should be ready to upload.")
        print("💡 Next: Create a GitHub repository and push your code.")
    else:
        print("⚠️  Some issues were found. Please fix them before uploading.")
        print("💡 See the solutions provided above.")
    
    suggest_next_steps()
    
    return all_good

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Troubleshooting cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please report this issue with the full error message")
