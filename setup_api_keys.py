#!/usr/bin/env python3
"""
Interactive API Key Setup

This script helps you set up your .env file with API keys interactively.
"""

import os
import sys
from pathlib import Path
from getpass import getpass


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print(" 🚀 AI Legal Analysis System - API Key Setup")
    print("=" * 60)
    print()
    print("This script will help you configure your API keys.")
    print("Your keys will be saved to the .env file.")
    print()


def get_api_key_info():
    """Return information about each API key."""
    return {
        'OPENAI_API_KEY': {
            'name': 'OpenAI',
            'description': 'Used for GPT-4 analysis and text embeddings',
            'signup_url': 'https://platform.openai.com/api-keys',
            'format': 'sk-proj-...',
            'required': True
        },
        'ANTHROPIC_API_KEY': {
            'name': 'Anthropic Claude',
            'description': 'Used for Claude 3.5 Sonnet analysis',
            'signup_url': 'https://console.anthropic.com/',
            'format': 'sk-ant-...',
            'required': True
        },
        'GEMINI_API_KEY': {
            'name': 'Google Gemini',
            'description': 'Used for Gemini 1.5 Flash analysis',
            'signup_url': 'https://makersuite.google.com/app/apikey',
            'format': 'AIza...',
            'required': True
        },
        'FIRECRAWL_API_KEY': {
            'name': 'Firecrawl',
            'description': 'Used for intelligent web scraping',
            'signup_url': 'https://firecrawl.dev/',
            'format': 'fc-...',
            'required': False
        }
    }


def load_existing_env():
    """Load existing .env file if it exists."""
    env_file = Path('.env')
    env_vars = {}
    
    if env_file.exists():
        print(f"📄 Found existing .env file: {env_file.absolute()}")
        
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
        
        print(f"📋 Loaded {len(env_vars)} existing environment variables")
    else:
        print("📄 No existing .env file found. Will create a new one.")
    
    return env_vars


def get_user_input(key_name, key_info, current_value=None):
    """Get API key from user input."""
    print(f"\n--- {key_info['name']} ---")
    print(f"Description: {key_info['description']}")
    print(f"Format: {key_info['format']}")
    print(f"Get your key at: {key_info['signup_url']}")
    
    if current_value and current_value != f"your-{key_name.lower().replace('_', '-')}-here":
        masked_value = f"{current_value[:8]}..." if len(current_value) > 8 else current_value
        print(f"Current value: {masked_value}")
        
        choice = input("Keep current value? [Y/n]: ").strip().lower()
        if choice in ['', 'y', 'yes']:
            return current_value
    
    if key_info['required']:
        print("⚠️  This API key is REQUIRED for the system to work properly.")
    else:
        print("ℹ️  This API key is optional but recommended.")
    
    while True:
        try:
            value = getpass(f"Enter your {key_info['name']} API key (input hidden): ").strip()
            
            if not value:
                if key_info['required']:
                    print("❌ This key is required. Please enter a value.")
                    continue
                else:
                    print("⏭️ Skipping optional key.")
                    return ""
            
            # Basic validation
            if len(value) < 10:
                print("❌ API key seems too short. Please check and try again.")
                continue
            
            # Format-specific validation
            format_prefix = key_info['format'].split('-')[0] + '-'
            if not value.startswith(format_prefix):
                print(f"⚠️  Warning: Key doesn't start with expected format '{format_prefix}'")
                confirm = input("Continue anyway? [y/N]: ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            
            print(f"✅ {key_info['name']} API key accepted!")
            return value
            
        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


def save_env_file(env_vars):
    """Save environment variables to .env file."""
    env_file = Path('.env')
    
    # Create backup if file exists
    if env_file.exists():
        backup_file = Path('.env.backup')
        backup_file.write_text(env_file.read_text())
        print(f"📁 Created backup: {backup_file.absolute()}")
    
    # Write new .env file
    with open(env_file, 'w') as f:
        f.write("# AI Legal Analysis System - Environment Configuration\n")
        f.write("# Generated by setup script\n\n")
        
        # API Keys section
        f.write("# =============================================================================\n")
        f.write("# EXTERNAL API KEYS\n")
        f.write("# =============================================================================\n\n")
        
        api_keys = get_api_key_info()
        for key_name, key_info in api_keys.items():
            value = env_vars.get(key_name, "")
            f.write(f"# {key_info['name']} - {key_info['description']}\n")
            f.write(f"{key_name}=\"{value}\"\n\n")
        
        # Other essential settings
        f.write("# =============================================================================\n")
        f.write("# APPLICATION CONFIGURATION\n")
        f.write("# =============================================================================\n\n")
        
        other_vars = {
            'DEBUG': env_vars.get('DEBUG', 'true'),
            'SECRET_KEY': env_vars.get('SECRET_KEY', 'your-secret-key-change-in-production'),
            'DATABASE_URL': env_vars.get('DATABASE_URL', 'postgresql+asyncpg://postgres:password@localhost:5432/ai_analysis'),
            'REDIS_URL': env_vars.get('REDIS_URL', 'redis://localhost:6379/0'),
            'LOG_LEVEL': env_vars.get('LOG_LEVEL', 'INFO'),
        }
        
        for key, value in other_vars.items():
            f.write(f"{key}=\"{value}\"\n")
    
    print(f"✅ Configuration saved to: {env_file.absolute()}")


def validate_setup():
    """Validate the setup by running the test script."""
    print("\n" + "=" * 60)
    print(" 🧪 Validating Configuration")
    print("=" * 60)
    
    try:
        # Import and run validation
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from config import get_settings, validate_configuration, print_configuration_status
        
        settings = get_settings()
        print_configuration_status()
        
        try:
            validate_configuration()
            print("\n✅ Configuration validation passed!")
            return True
        except ValueError as e:
            print(f"\n⚠️  Configuration warnings:")
            print(f"   {e}")
            print("   You can continue, but some features may not work.")
            return False
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        print("   Please check your configuration and try again.")
        return False


def main():
    """Main setup function."""
    print_banner()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working in: {Path.cwd()}")
    
    try:
        # Load existing environment
        env_vars = load_existing_env()
        
        # Get API key information
        api_keys = get_api_key_info()
        
        print("\n" + "=" * 60)
        print(" 🔑 API Key Configuration")
        print("=" * 60)
        
        # Collect API keys
        for key_name, key_info in api_keys.items():
            current_value = env_vars.get(key_name, "")
            new_value = get_user_input(key_name, key_info, current_value)
            env_vars[key_name] = new_value
        
        # Save configuration
        print("\n" + "=" * 60)
        print(" 💾 Saving Configuration")
        print("=" * 60)
        
        save_env_file(env_vars)
        
        # Validate setup
        validation_passed = validate_setup()
        
        # Final instructions
        print("\n" + "=" * 60)
        print(" 🎉 Setup Complete!")
        print("=" * 60)
        
        if validation_passed:
            print("\n✅ Your AI Legal Analysis System is ready to go!")
        else:
            print("\n⚠️  Setup completed with warnings. Check the issues above.")
        
        print("\nNext steps:")
        print("1. Start the system: docker-compose up -d")
        print("2. Access the API: http://localhost:8000/docs")
        print("3. Check health: curl http://localhost:8000/health")
        print("4. Upload documents and start analyzing!")
        
        print(f"\n📁 Your configuration is saved in: {Path.cwd() / '.env'}")
        print("📚 See README_COMPLETE.md for detailed usage instructions")
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
