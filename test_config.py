#!/usr/bin/env python3
"""
Test API Key Configuration

This script tests that API keys are properly loaded from the .env file
and displays the configuration status.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_settings, validate_configuration, print_configuration_status


def test_env_loading():
    """Test that environment variables are loaded correctly."""
    print("=== Testing .env File Loading ===")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✓ .env file found at: {env_file.absolute()}")
    else:
        print(f"✗ .env file not found. Looking for .env.example...")
        env_example = Path(".env.example")
        if env_example.exists():
            print(f"✓ .env.example found at: {env_example.absolute()}")
            print("  Please copy .env.example to .env and add your API keys")
        else:
            print("✗ Neither .env nor .env.example found!")
            return False
    
    print()
    
    # Load and test configuration
    try:
        settings = get_settings()
        print("✓ Settings loaded successfully")
        
        # Print configuration status
        print_configuration_status()
        
        # Test validation
        try:
            validate_configuration()
            print("✓ Configuration validation passed")
        except ValueError as e:
            print(f"⚠  Configuration validation warnings:")
            print(f"   {e}")
            print(f"   Note: This is expected if you haven't set all API keys yet")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        return False


def test_individual_keys():
    """Test individual API key loading."""
    print("\n=== Testing Individual API Key Loading ===")
    
    settings = get_settings()
    
    # Test direct environment variable access
    env_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY", ""),
    }
    
    print("Direct environment variable access:")
    for key, value in env_keys.items():
        status = "✓ Set" if value else "✗ Not set"
        masked = f"{value[:8]}..." if len(value) > 8 else "Not set"
        print(f"  {key}: {status} ({masked})")
    
    print("\nPydantic settings access:")
    pydantic_keys = {
        "openai_api_key": settings.openai_api_key,
        "anthropic_api_key": settings.anthropic_api_key,
        "gemini_api_key": settings.gemini_api_key,
        "firecrawl_api_key": settings.firecrawl_api_key,
    }
    
    for key, value in pydantic_keys.items():
        status = "✓ Set" if value else "✗ Not set"
        masked = f"{value[:8]}..." if len(value) > 8 else "Not set"
        print(f"  {key}: {status} ({masked})")
    
    # Check if they match
    matches = True
    for (env_key, env_val), (pyd_key, pyd_val) in zip(env_keys.items(), pydantic_keys.items()):
        if env_val != pyd_val:
            print(f"⚠  Mismatch: {env_key} != {pyd_key}")
            matches = False
    
    if matches:
        print("✓ Environment variables match Pydantic settings")
    
    return matches


def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_file = Path(".env")
    if not env_file.exists():
        print("\n=== Creating Sample .env File ===")
        
        sample_content = """# AI Legal Analysis System - Environment Configuration
# Copy this file to .env and add your actual API keys

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================
APP_NAME="AI Analysis System"
VERSION="1.0.0"
DEBUG=true
HOST="0.0.0.0"
PORT=8000

# Security
SECRET_KEY="your-secret-key-here-change-in-production"

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/ai_analysis"

# =============================================================================
# REDIS CONFIGURATION
# =============================================================================
REDIS_URL="redis://localhost:6379/0"

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================
# Add your actual API keys here:
OPENAI_API_KEY="your-openai-api-key-here"
ANTHROPIC_API_KEY="your-anthropic-api-key-here"
GEMINI_API_KEY="your-gemini-api-key-here"
FIRECRAWL_API_KEY="your-firecrawl-api-key-here"

# =============================================================================
# FILE STORAGE
# =============================================================================
UPLOAD_DIR="/tmp/uploads"
MAX_FILE_SIZE=104857600  # 100MB
"""
        
        env_file.write_text(sample_content)
        print(f"✓ Created sample .env file at: {env_file.absolute()}")
        print("  Please edit this file and add your actual API keys")
        return True
    
    return False


def main():
    """Run all tests."""
    print("API Key Configuration Test")
    print("=" * 50)
    
    # Change to the directory containing this script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {Path.cwd()}")
    
    # Create sample .env if needed
    create_sample_env()
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_env_loading():
        tests_passed += 1
    
    if test_individual_keys():
        tests_passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! API key configuration is working correctly.")
    else:
        print("⚠  Some tests failed. Please check your .env file configuration.")
        print("\nTroubleshooting:")
        print("1. Make sure .env file exists in the project root")
        print("2. Make sure API keys are set in .env file")
        print("3. Make sure .env file has no syntax errors")
        print("4. Make sure python-dotenv is installed: pip install python-dotenv")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
