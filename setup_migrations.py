# setup_migrations.py
"""
Setup script for initializing and running database migrations
"""

import os
import sys
import subprocess
from pathlib import Path
import asyncio
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rag_system.core.config import settings


def setup_alembic():
    """Initialize Alembic if not already set up"""
    migrations_dir = Path("src/rag_system/db/migrations")
    
    if not migrations_dir.exists():
        print("🔧 Initializing Alembic...")
        
        # Create migrations directory
        migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize alembic
        subprocess.run(["alembic", "init", str(migrations_dir)], check=True)
        
        # Create env.py for our setup
        env_py_content = '''"""Alembic environment script"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parents[4]))

from src.rag_system.db.models import Base
from src.rag_system.core.config import settings

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the database URL from settings
config.set_main_option('sqlalchemy.url', settings.database_url)

# Target metadata
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
        
        env_py_path = migrations_dir / "env.py"
        env_py_path.write_text(env_py_content)
        
        print("✅ Alembic initialized successfully!")
    else:
        print("ℹ️  Alembic already initialized")


def create_migration(message: str):
    """Create a new migration"""
    print(f"📝 Creating migration: {message}")
    subprocess.run(
        ["alembic", "revision", "--autogenerate", "-m", message],
        check=True
    )
    print("✅ Migration created successfully!")


def run_migrations():
    """Run pending migrations"""
    print("🚀 Running migrations...")
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    print("✅ Migrations completed successfully!")


def check_migration_status():
    """Check current migration status"""
    print("📊 Checking migration status...")
    subprocess.run(["alembic", "current"], check=True)
    print("\n📋 Migration history:")
    subprocess.run(["alembic", "history", "--verbose"], check=True)


async def setup_initial_embedding_models():
    """Set up initial embedding models in the database"""
    from src.rag_system.db.pgvector import db
    from src.rag_system.services.embeddings import embedding_service
    
    print("\n🎯 Setting up initial embedding models...")
    
    await db.initialize()
    
    # Get available models
    available_models = embedding_service.get_available_models()
    
    for model_name, dimension in available_models.items():
        try:
            model_id = await db.ensure_embedding_model(model_name, dimension)
            print(f"✅ Registered model: {model_name} (dimension={dimension})")
        except Exception as e:
            print(f"⚠️  Failed to register {model_name}: {e}")
    
    await db.close()


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration setup and management")
    parser.add_argument(
        "command",
        choices=["init", "create", "run", "status", "setup-models"],
        help="Command to execute"
    )
    parser.add_argument(
        "-m", "--message",
        help="Migration message (for create command)",
        default="Auto migration"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "init":
            setup_alembic()
        elif args.command == "create":
            create_migration(args.message)
        elif args.command == "run":
            run_migrations()
        elif args.command == "status":
            check_migration_status()
        elif args.command == "setup-models":
            asyncio.run(setup_initial_embedding_models())
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Command failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()