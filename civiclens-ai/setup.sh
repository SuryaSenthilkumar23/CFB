#!/bin/bash

# CivicLens AI - Setup Script
# This script sets up the development environment

set -e

echo "🛠️  Setting up CivicLens AI development environment..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p ai-microservice/models
mkdir -p backend/logs
mkdir -p frontend/dist

# Copy environment files if they don't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
fi

if [ ! -f frontend/.env ]; then
    echo "📝 Creating frontend/.env file..."
    cp frontend/.env.example frontend/.env
fi

if [ ! -f backend/.env ]; then
    echo "📝 Creating backend/.env file..."
    cp backend/.env.example backend/.env
fi

if [ ! -f ai-microservice/.env ]; then
    echo "📝 Creating ai-microservice/.env file..."
    cp ai-microservice/.env.example ai-microservice/.env
fi

# Install dependencies
echo "📦 Installing dependencies..."

# Frontend dependencies
echo "🌐 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Backend dependencies
echo "🖥️  Installing backend dependencies..."
cd backend
npm install
cd ..

# AI microservice dependencies
echo "🧠 Installing AI microservice dependencies..."
cd ai-microservice
pip install -r requirements.txt
cd ..

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Edit .env files with your API keys:"
echo "   - OPENROUTER_API_KEY"
echo "   - GOOGLE_MAPS_API_KEY"
echo ""
echo "2. Start the development servers:"
echo "   - For Docker: ./start.sh"
echo "   - For development:"
echo "     Terminal 1: cd frontend && npm run dev"
echo "     Terminal 2: cd backend && npm run dev"
echo "     Terminal 3: cd ai-microservice && python main.py"
echo ""
echo "3. Access the application:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:5000"
echo "   - AI Microservice: http://localhost:8000"