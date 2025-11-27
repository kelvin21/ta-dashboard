"""
Check if .env file is loaded correctly
"""
import os
from pathlib import Path

print("=" * 60)
print("Environment Variable Check")
print("=" * 60)

# Current working directory
print(f"\nCurrent working directory: {os.getcwd()}")

# Script directory
script_dir = Path(__file__).parent
print(f"Script directory: {script_dir}")

# Check .env file
env_path = script_dir / '.env'
print(f"\n.env file path: {env_path}")
print(f".env exists: {env_path.exists()}")

if env_path.exists():
    print(f".env file size: {env_path.stat().st_size} bytes")
    
    print("\n.env file contents (values masked):")
    with open(env_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    if 'URI' in key.upper() or 'PASSWORD' in key.upper():
                        print(f"  Line {line_num}: {key}=***MASKED*** (length: {len(value)})")
                    else:
                        print(f"  Line {line_num}: {key}={value}")

# Try loading with python-dotenv
print("\n" + "=" * 60)
print("Testing python-dotenv")
print("=" * 60)

try:
    from dotenv import load_dotenv
    print("✓ python-dotenv is installed")
    
    # Load from explicit path
    result = load_dotenv(dotenv_path=env_path, verbose=True)
    print(f"\nload_dotenv() returned: {result}")
    
    # Check what was loaded
    print("\nEnvironment variables after load_dotenv():")
    print(f"  USE_MONGODB: {os.getenv('USE_MONGODB', 'NOT SET')}")
    
    mongodb_uri = os.getenv('MONGODB_URI')
    if mongodb_uri:
        print(f"  MONGODB_URI: SET (length: {len(mongodb_uri)})")
        if len(mongodb_uri) > 40:
            print(f"  URI preview: {mongodb_uri[:20]}...{mongodb_uri[-20:]}")
    else:
        print(f"  MONGODB_URI: NOT SET")
    
    print(f"  MONGODB_DB_NAME: {os.getenv('MONGODB_DB_NAME', 'NOT SET')}")
    
    # Test DatabaseAdapter
    print("\n" + "=" * 60)
    print("Testing DatabaseAdapter")
    print("=" * 60)
    
    try:
        from db_adapter import DatabaseAdapter
        print("✓ db_adapter imported successfully")
        
        # This will trigger the connection
        print("\nAttempting to create DatabaseAdapter instance...")
        # db = DatabaseAdapter()  # Uncomment to test actual connection
        
    except Exception as e:
        print(f"❌ DatabaseAdapter error: {e}")
    
except ImportError:
    print("❌ python-dotenv not installed")
    print("   Run: pip install python-dotenv")

print("\n" + "=" * 60)
