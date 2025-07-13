# CivicLens AI - Civic Issue Reporting Platform

A full-stack AI-powered civic issue reporting platform that allows citizens to report civic issues using voice or text input. The system generates formal complaints, translates them to regional languages, and estimates ward information.

## 🚀 Features

- **Voice & Text Input**: Report issues using voice recordings or text descriptions
- **AI-Powered Processing**: Automatically generate formal complaints using LLM
- **Multi-language Support**: Translate complaints to Hindi, Tamil, and other regional languages
- **Ward Estimation**: Guess the municipal ward based on the provided address
- **PDF Export**: Download complaint reports as PDF documents
- **Responsive UI**: Modern, accessible interface built with React and Tailwind CSS

## 🏗️ Architecture

### Frontend (React + Vite)
- **React 18** with modern hooks and context
- **Tailwind CSS** for styling
- **i18next** for internationalization
- **Axios** for API communication
- **Voice recording** capabilities
- **PDF generation** with jsPDF

### Backend (Node.js + Express)
- **Express.js** REST API server
- **MongoDB** for data storage
- **Multer** for file uploads
- **Rate limiting** and security middleware
- **Input validation** and error handling

### AI Microservice (Python + FastAPI)
- **FastAPI** for high-performance API
- **OpenRouter** integration for LLM processing
- **Whisper** for speech-to-text conversion
- **Deep Translator** for language translation
- **Google Maps API** for ward estimation

## 🛠️ Setup Instructions

### Prerequisites
- Node.js 18+
- Python 3.8+
- MongoDB
- FFmpeg (for audio processing)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd civiclens-ai
```

### 2. Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env
# Edit .env with your configuration
npm run dev
```

### 3. Backend Setup
```bash
cd ../backend
npm install
cp .env.example .env
# Edit .env with your configuration
npm run dev
```

### 4. AI Microservice Setup
```bash
cd ../ai-microservice
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python main.py
```

## 📋 Environment Variables

### Frontend (.env)
```
VITE_API_URL=http://localhost:5000/api
VITE_APP_NAME=CivicLens AI
VITE_APP_VERSION=1.0.0
```

### Backend (.env)
```
NODE_ENV=development
PORT=5000
MONGODB_URI=mongodb://localhost:27017/civiclens-ai
AI_SERVICE_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
LOG_LEVEL=info
```

### AI Microservice (.env)
```
HOST=0.0.0.0
PORT=8000
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=mistralai/mixtral-8x7b-instruct
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
WHISPER_MODEL=base
LOG_LEVEL=INFO
```

## 🔑 API Keys Required

1. **OpenRouter API Key**: Get from [OpenRouter](https://openrouter.ai/)
2. **Google Maps API Key**: Get from [Google Cloud Console](https://console.cloud.google.com/)

## 📁 Project Structure

```
civiclens-ai/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/          # Custom hooks
│   │   ├── utils/          # Utility functions
│   │   ├── styles/         # CSS styles
│   │   └── locales/        # Translation files
│   ├── public/             # Static assets
│   └── package.json
├── backend/                 # Node.js backend
│   ├── src/
│   │   ├── routes/         # API routes
│   │   ├── models/         # Database models
│   │   ├── middleware/     # Express middleware
│   │   └── utils/          # Utility functions
│   ├── config/             # Configuration files
│   └── package.json
├── ai-microservice/         # Python AI service
│   ├── src/
│   │   ├── services/       # AI processing services
│   │   ├── models/         # Pydantic models
│   │   └── utils/          # Utility functions
│   ├── requirements.txt
│   └── main.py
└── README.md
```

## 🔄 API Endpoints

### Backend API (`/api`)
- `POST /submit` - Submit a complaint
- `POST /generate-complaint` - Generate formal complaint
- `POST /voice-to-text` - Convert voice to text
- `POST /translate` - Translate text
- `GET /complaints` - Get complaints (with pagination)
- `GET /complaints/:id` - Get specific complaint
- `PUT /complaints/:id/status` - Update complaint status

### AI Microservice API
- `POST /voice-to-text` - Convert audio to text
- `POST /generate-complaint` - Generate formal complaint
- `POST /translate` - Translate text
- `POST /estimate-ward` - Estimate ward from address
- `GET /health` - Health check

## 🎯 Usage Flow

1. **User Input**: User describes the civic issue via voice or text
2. **Voice Processing**: If voice input, convert to text using Whisper
3. **Complaint Generation**: Generate formal complaint using OpenRouter LLM
4. **Translation**: Translate complaint to selected regional language
5. **Ward Estimation**: Estimate ward information using Google Maps API
6. **Storage**: Save complaint to MongoDB
7. **Results**: Display formatted results with download options

## 🌐 Supported Languages

- **English** (en)
- **Hindi** (hi) - हिंदी
- **Tamil** (ta) - தமிழ்

## 🔧 Development

### Running in Development Mode
```bash
# Terminal 1 - Frontend
cd frontend && npm run dev

# Terminal 2 - Backend
cd backend && npm run dev

# Terminal 3 - AI Microservice
cd ai-microservice && python main.py
```

### Building for Production
```bash
# Frontend
cd frontend && npm run build

# Backend
cd backend && npm run start

# AI Microservice
cd ai-microservice && uvicorn main:app --host 0.0.0.0 --port 8000
```

## 📊 Monitoring & Logging

- **Winston** logging in backend
- **Python logging** in AI microservice
- **Error tracking** with detailed stack traces
- **Performance monitoring** with request timing

## 🔒 Security Features

- **Rate limiting** on API endpoints
- **Input validation** and sanitization
- **CORS** protection
- **Helmet** security headers
- **File upload** restrictions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenRouter for LLM API access
- OpenAI Whisper for speech recognition
- Google Maps API for geolocation services
- Deep Translator for translation services

## 📞 Support

For support, please open an issue in the repository or contact the development team.

---

**CivicLens AI** - Empowering citizens through AI-powered civic engagement.