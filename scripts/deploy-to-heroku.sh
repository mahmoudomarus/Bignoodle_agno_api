#!/bin/bash

# Heroku deployment script
APP_NAME=${1:-"agno-agent-api-tabi"}

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "Heroku CLI not found. Please install it first."
    exit 1
fi

echo "Deploying to Heroku app: $APP_NAME"

# Build and push using Docker
echo "Logging in to Heroku Container Registry..."
heroku container:login

echo "Building Docker image..."
heroku container:push web --app $APP_NAME

echo "Releasing Docker image..."
heroku container:release web --app $APP_NAME

echo "Deployment complete. Your API should be available at:"
echo "https://$APP_NAME.herokuapp.com/"
echo ""
echo "Check the API documentation at:"
echo "https://$APP_NAME.herokuapp.com/docs" 