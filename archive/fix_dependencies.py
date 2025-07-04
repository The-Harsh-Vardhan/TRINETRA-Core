import subprocess
import sys

def run_pip_command(command):
    try:
        subprocess.check_call([sys.executable, "-m", "pip"] + command.split())
        return True
    except subprocess.CalledProcessError:
        print(f"Error executing: pip {command}")
        return False

def fix_dependencies():
    # First uninstall all conflicting packages
    packages_to_remove = [
        "opencv-python",
        "numpy",
        "scipy",
        "torch",
        "torchvision"
    ]
    
    print("Uninstalling current versions...")
    for package in packages_to_remove:
        run_pip_command(f"uninstall {package} -y")

    # Install specific versions in order (numpy first as it's a common dependency)
    packages_to_install = [
        "numpy==1.26.0",
        "scipy==1.11.1",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "opencv-python==4.8.0.74"
    ]

    print("\nInstalling required versions...")
    for package in packages_to_install:
        print(f"\nInstalling {package}...")
        if not run_pip_command(f"install {package} --no-deps"):
            print(f"Failed to install {package}")
            continue
        
    # Finally, install dependencies
    print("\nInstalling dependencies...")
    run_pip_command("install --no-deps -r requirements.txt")
    run_pip_command("install -r requirements.txt")

if __name__ == "__main__":
    fix_dependencies()
