#!/bin/bash

# CivicLens AI - Startup Script
# This script starts all services in the correct order

set -e

echo "🚀 Starting CivicLens AI Platform..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys and configuration."
    echo "   Required: OPENROUTER_API_KEY, GOOGLE_MAPS_API_KEY"
    exit 1
fi

# Load environment variables
source .env

# Check required environment variables
if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "your_openrouter_api_key_here" ]; then
    echo "❌ OPENROUTER_API_KEY is not set in .env file"
    exit 1
fi

if [ -z "$GOOGLE_MAPS_API_KEY" ] || [ "$GOOGLE_MAPS_API_KEY" = "your_google_maps_api_key_here" ]; then
    echo "❌ GOOGLE_MAPS_API_KEY is not set in .env file"
    exit 1
fi

# Start services
echo "🔧 Starting services..."

# Start MongoDB first
echo "📦 Starting MongoDB..."
docker-compose up -d mongodb

# Wait for MongoDB to be ready
echo "⏳ Waiting for MongoDB to be ready..."
until docker-compose exec mongodb mongo --eval "print('MongoDB is ready')" &> /dev/null; do
    sleep 2
done

# Start AI microservice
echo "🧠 Starting AI microservice..."
docker-compose up -d ai-microservice

# Wait for AI microservice to be ready
echo "⏳ Waiting for AI microservice to be ready..."
until curl -f http://localhost:8000/health &> /dev/null; do
    sleep 2
done

# Start backend
echo "🖥️  Starting backend..."
docker-compose up -d backend

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
until curl -f http://localhost:5000/health &> /dev/null; do
    sleep 2
done

# Start frontend
echo "🌐 Starting frontend..."
docker-compose up -d frontend

# Wait for frontend to be ready
echo "⏳ Waiting for frontend to be ready..."
until curl -f http://localhost:3000/health &> /dev/null; do
    sleep 2
done

echo "✅ All services are running!"
echo ""
echo "🌟 CivicLens AI is ready!"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:5000"
echo "   AI Microservice: http://localhost:8000"
echo "   MongoDB: mongodb://localhost:27017"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f [service-name]"
echo ""
echo "🛑 To stop all services:"
echo "   docker-compose down"
echo ""
echo "🔄 To restart services:"
echo "   docker-compose restart [service-name]"