import pkg_resources
import sys

def check_requirements():
    """Check if all required packages are installed with correct versions"""
    required = {
        'opencv-python': '4.8.0.74',
        'scipy': '1.11.1',
        'torch': '2.0.1',
        'torchvision': '0.15.2',
        'motor': '3.1.1',
        'pymongo': '4.3.3',
        'fastapi': '0.100.0',
        'uvicorn': '0.22.0',
        'python-multipart': '0.0.6',
        'deepface': '0.0.79',
        'ultralytics': '8.0.124',
        'supervision': '0.11.1'
    }
    
    missing = []
    version_mismatch = []
    
    for package, version in required.items():
        try:
            installed = pkg_resources.get_distribution(package)
            if installed.version != version:
                version_mismatch.append(
                    f"⚠️ {package} version mismatch (required: {version}, installed: {installed.version})"
                )
        except pkg_resources.DistributionNotFound:
            missing.append(f"❌ {package} is not installed")
    
    if missing or version_mismatch:
        print("\nPackage Requirements Check:")
        if missing:
            print("\nMissing Packages:")
            for msg in missing:
                print(msg)
        if version_mismatch:
            print("\nVersion Mismatches:")
            for msg in version_mismatch:
                print(msg)
        return False
    
    print("✅ All required packages are installed with correct versions")
    return True

if __name__ == "__main__":
    sys.exit(0 if check_requirements() else 1)
