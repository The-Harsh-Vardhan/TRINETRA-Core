import pkg_resources
import subprocess
import sys
import re

def fix_version_mismatches():
    try:
        # Read requirements
        with open('requirements.txt', 'r') as file:
            requirements = [line.split('#')[0].strip() 
                          for line in file 
                          if line.split('#')[0].strip()]

        print("Checking for version mismatches...")
        to_reinstall = []

        for requirement in requirements:
            match = re.match(r'([^=<>]+)([=<>]+.+)?', requirement)
            if not match:
                continue
                
            package_name = match.group(1).strip()
            version_spec = match.group(2) if match.group(2) else None

            try:
                installed_package = pkg_resources.get_distribution(package_name)
                if version_spec:
                    required_version = version_spec.replace('==', '').strip()
                    if str(installed_package.version) != required_version:
                        to_reinstall.append(requirement)
            except pkg_resources.DistributionNotFound:
                to_reinstall.append(requirement)

        if to_reinstall:
            print("\nReinstalling packages with version mismatches:")
            for package in to_reinstall:
                print(f"Reinstalling {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-I", package])
                    print(f"Successfully reinstalled {package}")
                except subprocess.CalledProcessError:
                    print(f"Failed to reinstall {package}")
        else:
            print("No version mismatches found!")

    except FileNotFoundError:
        print("Error: requirements.txt not found in current directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    fix_version_mismatches()
