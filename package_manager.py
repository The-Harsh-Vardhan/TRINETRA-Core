#!/usr/bin/env python3
"""
Unified Package Management Utility for TRINETRA-Core
Handles checking, installing, and fixing package dependencies.
"""

import pkg_resources
import subprocess
import sys
import re
from typing import Dict, List, Tuple, Optional

class PackageManager:
    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = requirements_file
        self.required_packages = self._parse_requirements()
    
    def _parse_requirements(self) -> Dict[str, str]:
        """Parse requirements.txt file"""
        packages = {}
        try:
            with open(self.requirements_file, 'r') as file:
                for line in file:
                    line = line.split('#')[0].strip()
                    if line and '==' in line:
                        name, version = line.split('==')
                        packages[name.strip()] = version.strip()
        except FileNotFoundError:
            print(f"❌ {self.requirements_file} not found")
        return packages
    
    def _run_pip_command(self, command: List[str]) -> bool:
        """Execute pip command safely"""
        try:
            subprocess.check_call([sys.executable, "-m", "pip"] + command)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error executing pip command: {e}")
            return False
    
    def check_packages(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Check package status
        Returns: (correctly_installed, version_mismatches, missing)
        """
        correctly_installed = []
        version_mismatches = []
        missing = []
        
        for package, required_version in self.required_packages.items():
            try:
                installed = pkg_resources.get_distribution(package)
                if installed.version == required_version:
                    correctly_installed.append(f"✅ {package}=={required_version}")
                else:
                    version_mismatches.append(
                        f"⚠️ {package} (required: {required_version}, installed: {installed.version})"
                    )
            except pkg_resources.DistributionNotFound:
                missing.append(f"❌ {package}=={required_version}")
        
        return correctly_installed, version_mismatches, missing
    
    def install_packages(self, package_list: Optional[List[str]] = None) -> bool:
        """Install packages from requirements or specific list"""
        if package_list is None:
            return self._run_pip_command(["install", "-r", self.requirements_file])
        else:
            success = True
            for package in package_list:
                if not self._run_pip_command(["install", package]):
                    success = False
            return success
    
    def fix_versions(self) -> bool:
        """Fix version mismatches"""
        _, mismatches, missing = self.check_packages()
        
        if not mismatches and not missing:
            print("✅ All packages are correctly installed")
            return True
        
        # Install missing packages
        if missing:
            print("Installing missing packages...")
            for package_info in missing:
                package = package_info.replace("❌ ", "")
                if not self._run_pip_command(["install", package]):
                    return False
        
        # Fix version mismatches
        if mismatches:
            print("Fixing version mismatches...")
            for mismatch in mismatches:
                package_name = mismatch.split(' ')[1]
                required_version = self.required_packages[package_name]
                package_spec = f"{package_name}=={required_version}"
                if not self._run_pip_command(["install", package_spec, "--force-reinstall"]):
                    return False
        
        return True
    
    def uninstall_and_reinstall(self) -> bool:
        """Complete reinstall of all packages"""
        print("🔄 Performing complete package reinstall...")
        
        # Uninstall problematic packages first
        problematic_packages = ["torch", "torchvision", "opencv-python"]
        for package in problematic_packages:
            if package in self.required_packages:
                print(f"Uninstalling {package}...")
                self._run_pip_command(["uninstall", package, "-y"])
        
        # Reinstall all packages
        return self.install_packages()
    
    def print_status(self):
        """Print package status report"""
        correct, mismatches, missing = self.check_packages()
        
        print("\n" + "="*50)
        print("📦 PACKAGE STATUS REPORT")
        print("="*50)
        
        if correct:
            print("\n✅ Correctly Installed:")
            for package in correct:
                print(f"  {package}")
        
        if mismatches:
            print("\n⚠️ Version Mismatches:")
            for package in mismatches:
                print(f"  {package}")
        
        if missing:
            print("\n❌ Missing Packages:")
            for package in missing:
                print(f"  {package}")
        
        if not mismatches and not missing:
            print("\n🎉 All packages are correctly installed!")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TRINETRA Package Manager")
    parser.add_argument("action", choices=["check", "install", "fix", "reinstall"],
                       help="Action to perform")
    parser.add_argument("--requirements", default="requirements.txt",
                       help="Requirements file path")
    
    args = parser.parse_args()
    
    manager = PackageManager(args.requirements)
    
    if args.action == "check":
        manager.print_status()
    elif args.action == "install":
        if manager.install_packages():
            print("✅ Installation completed successfully")
        else:
            print("❌ Installation failed")
    elif args.action == "fix":
        if manager.fix_versions():
            print("✅ Package versions fixed successfully")
        else:
            print("❌ Failed to fix package versions")
    elif args.action == "reinstall":
        if manager.uninstall_and_reinstall():
            print("✅ Complete reinstall completed successfully")
        else:
            print("❌ Reinstall failed")

if __name__ == "__main__":
    main()
