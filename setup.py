#!/usr/bin/env python3
"""
Stock Market Analytics Setup
============================

This script sets up your MongoDB-integrated stock analytics project.

Usage:
    python setup.py
"""

import os
import sys
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def create_env_file():
    """Create .env file with MongoDB configuration."""
    print("🔧 Creating .env file...")
    
    # Check if .env already exists
    if os.path.exists(".env"):
        print("⚠️  .env file already exists")
        overwrite = input("Overwrite it? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Using existing .env file")
            return True
    
    # Get MongoDB URI from user
    print("\nMongoDB Atlas Connection Setup")
    print("=" * 30)
    print("Enter your MongoDB Atlas connection string:")
    print("Example: mongodb+srv://username:password@cluster.mongodb.net/")
    print()
    
    uri = input("MongoDB URI: ").strip()
    if not uri:
        print("❌ No URI provided")
        return False
    
    db_name = input("Database name (default: sectorscope): ").strip() or "sectorscope"
    
    # Create .env content
    env_content = f"""# MongoDB Configuration
MONGODB_URI={uri}
MONGODB_DB={db_name}

# Optional: Cache settings
CACHE_TTL_MIN=30

# Optional: Logging level
LOG_LEVEL=INFO

# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def test_connection():
    """Test MongoDB connection."""
    print("\n🔍 Testing MongoDB connection...")
    
    try:
        from pymongo import MongoClient
        
        uri = os.environ.get("MONGODB_URI", "").strip()
        if not uri:
            print("❌ MONGODB_URI not found in .env file")
            return False
        
        print(f"Connecting to: {uri[:50]}...")
        
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        client.admin.command("ping")
        print("✅ Connection successful!")
        
        # Test database access
        db_name = os.environ.get("MONGODB_DB", "sectorscope")
        db = client[db_name]
        collections = db.list_collection_names()
        print(f"✅ Database '{db_name}' accessible ({len(collections)} collections)")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check MongoDB Atlas cluster is running")
        print("2. Verify IP address is whitelisted (0.0.0.0/0)")
        print("3. Check username/password are correct")
        return False

def setup_database():
    """Set up MongoDB database with schema and data."""
    print("\n🚀 Setting up MongoDB database...")
    
    try:
        from scripts.setup_mongodb_atlas import MongoDBAtlasSetup
        from scripts.mongodb_utils import MongoDBContext
        
        with MongoDBContext() as mongo_manager:
            setup_tool = MongoDBAtlasSetup(mongo_manager)
            
            success = setup_tool.run_complete_setup(
                migrate_csv=True,
                download_fresh=True
            )
            
            if success:
                print("✅ Database setup completed!")
                return True
            else:
                print("❌ Database setup failed")
                return False
                
    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False

def main():
    """Main setup function."""
    print("Stock Market Analytics Setup")
    print("=" * 30)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Create .env file
    if not create_env_file():
        print("❌ Setup failed at .env creation")
        return
    
    # Step 2: Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("❌ python-dotenv not installed")
        print("Please run: pip install python-dotenv")
        return
    
    # Step 3: Test connection
    if not test_connection():
        print("❌ Setup failed at connection test")
        return
    
    # Step 4: Setup database
    if not setup_database():
        print("❌ Setup failed at database setup")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the application: python main.py")
    print("2. Check database status: python scripts/mongodb_dashboard.py")
    print("3. View web interface: python web/app.py")

if __name__ == "__main__":
    main()
