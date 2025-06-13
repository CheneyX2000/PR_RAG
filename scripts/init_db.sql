-- scripts/init-db.sql
-- Database initialization script for RAG system

-- Create pgvector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Set default configurations for better performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET effective_cache_size = '1GB';

-- Log for debugging
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully';
    RAISE NOTICE 'PgVector extension version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END $$;