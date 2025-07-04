import subprocess
import sys

def install_requirements():
    try:
        # Read requirements.txt
        with open('requirements.txt', 'r') as file:
            # Split on '#' and take first part, then strip whitespace
            requirements = [line.split('#')[0].strip() 
                          for line in file 
                          if line.split('#')[0].strip()]

        print("Starting installation of packages...")
        
        # Install each package
        for package in requirements:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}")
                continue

        print("\nInstallation process completed!")

    except FileNotFoundError:
        print("Error: requirements.txt not found in current directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    install_requirements()
