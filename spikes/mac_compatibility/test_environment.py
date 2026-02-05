#!/usr/bin/env python3
"""
Test 1: Environment Setup Verification

Checks that the macOS environment has all prerequisites:
- Python version
- Java installation
- Required packages
- Configuration files
"""

import sys
import subprocess
import os
from pathlib import Path


def test_python_version():
    """Check Python version is 3.x."""
    print("\n[1/5] Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3:
        print("  ✓ Python 3.x found")
        return True
    else:
        print("  ✗ Python 3.x required")
        return False


def test_java_installation():
    """Check if Java is installed."""
    print("\n[2/5] Checking Java installation...")
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_output = result.stderr.split("\n")[0]
        print(f"  {version_output}")
        print("  ✓ Java found")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ✗ Java not found or not responding")
        print("  Install Java: brew install openjdk")
        return False


def test_dependencies():
    """Check if required packages are installed."""
    print("\n[3/5] Checking Python dependencies...")

    packages = [
        ("google-generativeai", "google.generativeai"),
        ("jpype1", "jpype"),
        ("python-dotenv", "dotenv"),
        ("requests", "requests"),
        ("pytest", "pytest"),
    ]

    all_found = True
    for pkg_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"  ✓ {pkg_name}")
        except ImportError:
            print(f"  ✗ {pkg_name} - NOT INSTALLED")
            all_found = False

    if not all_found:
        print("\n  Install missing packages:")
        print("  pip install -r requirements.txt")

    return all_found


def test_env_file():
    """Check if .env file exists and has required keys."""
    print("\n[4/5] Checking .env configuration...")

    env_path = Path(__file__).parent.parent.parent / ".env"

    if not env_path.exists():
        print(f"  ✗ .env file not found at {env_path}")
        print("\n  Create .env file with:")
        print("  GOOGLE_API_KEY=your_api_key_here")
        print("  AUTOPLOT_JAR=/path/to/autoplot.jar")
        return False

    print(f"  ✓ .env file exists")

    # Check for required keys
    from dotenv import load_dotenv
    load_dotenv(env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    jar_path = os.getenv("AUTOPLOT_JAR")

    results = []
    if api_key:
        print(f"  ✓ GOOGLE_API_KEY is set")
        results.append(True)
    else:
        print(f"  ✗ GOOGLE_API_KEY not set")
        results.append(False)

    if jar_path:
        print(f"  ✓ AUTOPLOT_JAR is set: {jar_path}")
        results.append(True)
    else:
        print(f"  ✗ AUTOPLOT_JAR not set")
        results.append(False)

    return all(results)


def test_autoplot_jar():
    """Check if Autoplot JAR exists at configured path."""
    print("\n[5/5] Checking Autoplot JAR file...")

    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    jar_path_str = os.getenv("AUTOPLOT_JAR")
    if not jar_path_str:
        print("  ✗ AUTOPLOT_JAR not configured in .env")
        return False

    jar_path = Path(jar_path_str).expanduser().resolve()

    if jar_path.exists():
        size_mb = jar_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ JAR file found: {jar_path}")
        print(f"    Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ✗ JAR file not found: {jar_path}")
        print("\n  Download Autoplot:")
        print("  wget -O ~/autoplot.jar https://autoplot.org/jnlp/latest/autoplot.jar")
        return False


def main():
    """Run all environment tests."""
    print("=" * 60)
    print("macOS Environment Verification")
    print("=" * 60)

    results = [
        test_python_version(),
        test_java_installation(),
        test_dependencies(),
        test_env_file(),
        test_autoplot_jar(),
    ]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if all(results):
        print("\n✓ Environment is ready for testing!")
        return 0
    else:
        print("\n✗ Some checks failed. Fix issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
