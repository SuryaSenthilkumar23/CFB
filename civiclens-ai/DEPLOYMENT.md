# CivicLens AI - Deployment Guide

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# 1. Clone and setup
git clone <repository>
cd civiclens-ai
./setup.sh

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your API keys

# 3. Start all services
./start.sh
```

### Option 2: Development Mode
```bash
# Terminal 1 - Frontend
cd frontend && npm run dev

# Terminal 2 - Backend  
cd backend && npm run dev

# Terminal 3 - AI Microservice
cd ai-microservice && python main.py
```

## 📋 Prerequisites

### Required
- Docker & Docker Compose (for containerized deployment)
- Node.js 18+ (for development)
- Python 3.8+ (for AI microservice)
- MongoDB (local or cloud)

### API Keys Required
- **OpenRouter API Key**: [Get from OpenRouter](https://openrouter.ai/)
- **Google Maps API Key**: [Get from Google Cloud Console](https://console.cloud.google.com/)

## 🔧 Configuration

### Environment Variables
Update `.env` with your configuration:
```env
OPENROUTER_API_KEY=your_actual_key_here
GOOGLE_MAPS_API_KEY=your_actual_key_here
WHISPER_MODEL=base
MONGODB_URI=mongodb://localhost:27017/civiclens-ai
```

### Service Ports
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:5000`
- AI Microservice: `http://localhost:8000`
- MongoDB: `mongodb://localhost:27017`

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │ AI Microservice │
│   (React)       │◄──►│   (Node.js)     │◄──►│   (Python)      │
│   Port: 3000    │    │   Port: 5000    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   MongoDB       │
                       │   Port: 27017   │
                       └─────────────────┘
```

## 🧪 Testing

### Frontend Testing
```bash
cd frontend
npm test
```

### Backend Testing
```bash
cd backend
npm test
```

### API Testing
```bash
# Health checks
curl http://localhost:3000/health  # Frontend
curl http://localhost:5000/health  # Backend
curl http://localhost:8000/health  # AI Service
```

## 🔄 API Endpoints

### Backend API (`/api`)
- `POST /submit` - Submit complaint
- `POST /generate-complaint` - Generate formal complaint
- `POST /voice-to-text` - Convert voice to text
- `POST /translate` - Translate text
- `GET /complaints` - Get complaints
- `GET /complaints/:id` - Get specific complaint

### AI Microservice
- `POST /voice-to-text` - Audio to text conversion
- `POST /generate-complaint` - Generate complaint
- `POST /translate` - Text translation
- `POST /estimate-ward` - Ward estimation

## 📊 Monitoring

### Docker Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f ai-microservice
```

### Health Monitoring
All services include health check endpoints:
- Frontend: `/health`
- Backend: `/health`
- AI Service: `/health`

## 🔒 Security

- Rate limiting on API endpoints
- Input validation and sanitization
- CORS protection
- Security headers with Helmet
- File upload restrictions
- Non-root Docker containers

## 🛠️ Troubleshooting

### Common Issues

1. **API Keys Not Working**
   - Verify keys are correctly set in `.env`
   - Check API key permissions and quotas

2. **MongoDB Connection Failed**
   - Ensure MongoDB is running
   - Check connection string in `.env`

3. **Voice Processing Failed**
   - Verify FFmpeg is installed
   - Check audio file format support

4. **Docker Services Not Starting**
   - Run `docker-compose logs [service]`
   - Check port conflicts
   - Verify Docker daemon is running

### Debug Commands
```bash
# Check service status
docker-compose ps

# Restart specific service
docker-compose restart backend

# Rebuild service
docker-compose up --build backend

# Stop all services
docker-compose down
```

## 📈 Performance Optimization

### Production Recommendations
- Use environment-specific configurations
- Enable Redis for caching
- Implement CDN for static assets
- Use production-grade MongoDB setup
- Configure load balancing for high traffic

### Scaling
- Use Docker Swarm or Kubernetes for orchestration
- Implement horizontal pod autoscaling
- Use message queues for async processing
- Configure database replication

## 🔄 Updates and Maintenance

### Regular Updates
```bash
# Update dependencies
cd frontend && npm update
cd backend && npm update
cd ai-microservice && pip install -r requirements.txt --upgrade

# Rebuild containers
docker-compose build --no-cache
```

### Database Maintenance
- Regular backup of MongoDB
- Monitor database performance
- Update indexes as needed

## 🆘 Support

For issues and support:
1. Check logs for error messages
2. Review environment configuration
3. Verify API key validity
4. Check network connectivity
5. Open issue in repository

---

**Happy Deploying! 🚀**