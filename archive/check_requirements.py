import pkg_resources
import re
from pkg_resources import parse_version

def check_requirements():
    try:
        # Read requirements
        with open('requirements.txt', 'r') as file:
            requirements = [line.split('#')[0].strip() 
                          for line in file 
                          if line.split('#')[0].strip()]

        print("Checking installed packages...")
        missing = []
        version_mismatch = []
        installed = []

        for requirement in requirements:
            # Parse package name and version
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
                        version_mismatch.append(f"{package_name} (required: {required_version}, installed: {installed_package.version})")
                    else:
                        installed.append(f"{package_name}=={installed_package.version}")
                else:
                    installed.append(f"{package_name}=={installed_package.version}")
                    
            except pkg_resources.DistributionNotFound:
                missing.append(package_name)

        # Print results
        print("\nResults:")
        if installed:
            print("\nCorrectly installed packages:")
            for pkg in installed:
                print(f"✓ {pkg}")
                
        if version_mismatch:
            print("\nVersion mismatches:")
            for pkg in version_mismatch:
                print(f"⚠ {pkg}")
                
        if missing:
            print("\nMissing packages:")
            for pkg in missing:
                print(f"✗ {pkg}")

    except FileNotFoundError:
        print("Error: requirements.txt not found in current directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    check_requirements()
