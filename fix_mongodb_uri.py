"""
Fix MongoDB URI by properly encoding username and password.
"""
from urllib.parse import quote_plus

def fix_mongodb_uri():
    """Guide user through fixing MongoDB URI with special characters."""
    
    print("=" * 60)
    print("MongoDB URI Fixer")
    print("=" * 60)
    print("\nIf your password has special characters like @, #, $, %, &, etc.")
    print("they need to be URL-encoded.\n")
    
    print("Enter your MongoDB details:")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    cluster = input("Cluster URL (e.g., cluster0.abcde.mongodb.net): ").strip()
    
    # Encode username and password
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)
    
    # Build URI
    uri = f"mongodb+srv://{encoded_username}:{encoded_password}@{cluster}/?retryWrites=true&w=majority"
    
    print("\n" + "=" * 60)
    print("✅ Fixed MongoDB URI:")
    print("=" * 60)
    print(uri)
    print("\n" + "=" * 60)
    print("\nAdd this to your .env file:")
    print("=" * 60)
    print(f"MONGODB_URI={uri}")
    print("\nOr set as environment variable (PowerShell):")
    print(f'$env:MONGODB_URI="{uri}"')
    print("\n" + "=" * 60)
    
    return uri

if __name__ == "__main__":
    fix_mongodb_uri()
