import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  // Submit complaint
  submitComplaint: (complaintData) => {
    return api.post('/submit', complaintData);
  },

  // Convert voice to text
  voiceToText: (audioFile) => {
    const formData = new FormData();
    formData.append('audio', audioFile);
    
    return api.post('/voice-to-text', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Generate formal complaint
  generateComplaint: (data) => {
    return api.post('/generate-complaint', data);
  },

  // Translate text
  translateText: (data) => {
    return api.post('/translate', data);
  },

  // Get complaint history
  getComplaintHistory: () => {
    return api.get('/complaints');
  },

  // Get complaint by ID
  getComplaintById: (id) => {
    return api.get(`/complaints/${id}`);
  },

  // Update complaint status
  updateComplaintStatus: (id, status) => {
    return api.put(`/complaints/${id}/status`, { status });
  },
};

export default api;