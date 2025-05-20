#!/bin/bash

# This script extracts database connection details from the Heroku DATABASE_URL
# and sets them as environment variables expected by the application

# Extract database connection details from DATABASE_URL
# Format: postgres://username:password@host:port/database
if [ -n "$DATABASE_URL" ]; then
  echo "Parsing DATABASE_URL for database connection settings..."
  
  # Extract components from DATABASE_URL
  DB_DRIVER=$(echo $DATABASE_URL | sed -E 's/^([^:]+):\/\/.*/\1/')
  DB_USER=$(echo $DATABASE_URL | sed -E 's/^[^:]+:\/\/([^:]+):.*/\1/')
  DB_PASS=$(echo $DATABASE_URL | sed -E 's/^[^:]+:\/\/[^:]+:([^@]+)@.*/\1/')
  DB_HOST=$(echo $DATABASE_URL | sed -E 's/^[^:]+:\/\/[^@]+@([^:]+):.*/\1/')
  DB_PORT=$(echo $DATABASE_URL | sed -E 's/^[^:]+:\/\/[^@]+@[^:]+:([^\/]+)\/.*/\1/')
  DB_DATABASE=$(echo $DATABASE_URL | sed -E 's/^[^:]+:\/\/[^@]+@[^:]+:[^\/]+\/([^?]+).*/\1/')
  
  # Fix: Use postgresql instead of postgres+psycopg since that's causing issues
  export DB_DRIVER="postgresql"
  export DB_USER="${DB_USER}"
  export DB_PASS="${DB_PASS}"
  export DB_HOST="${DB_HOST}"
  export DB_PORT="${DB_PORT}"
  export DB_DATABASE="${DB_DATABASE}"
  
  echo "Database connection settings configured successfully."
else
  echo "DATABASE_URL not found. Please set up the database add-on in Heroku."
  exit 1
fi

# Start the application
exec uvicorn api.main:app --host 0.0.0.0 --port $PORT 