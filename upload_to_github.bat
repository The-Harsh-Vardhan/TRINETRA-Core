@echo off
echo Starting TRINETRA-Core GitHub Upload Helper...
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

:: Check if we're in the right directory
if not exist "github_upload_helper.py" (
    echo Error: github_upload_helper.py not found
    echo Please run this script from the TRINETRA-Core directory
    pause
    exit /b 1
)

:: Run the Python script
echo Running GitHub upload helper...
python github_upload_helper.py

:: Check if the script succeeded
if %errorlevel% neq 0 (
    echo.
    echo Upload failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Upload completed successfully!
echo Your project should now be available on GitHub.
echo.
pause
