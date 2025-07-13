import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Send, Mic, Type, AlertCircle } from 'lucide-react';
import VoiceRecorder from './VoiceRecorder';
import { apiService } from '../utils/api';

const ComplaintForm = ({ onSubmitSuccess }) => {
  const { t } = useTranslation();
  const [formData, setFormData] = useState({
    description: '',
    address: '',
    category: 'other',
    language: 'en',
  });
  const [inputMode, setInputMode] = useState('text'); // 'text' or 'voice'
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [voiceBlob, setVoiceBlob] = useState(null);

  const categories = [
    'pothole',
    'garbage',
    'streetlight',
    'water',
    'drainage',
    'other',
  ];

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'हिंदी' },
    { code: 'ta', name: 'தமிழ்' },
  ];

  const handleInputChange = useCallback((e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
  }, []);

  const handleVoiceRecording = useCallback((blob) => {
    setVoiceBlob(blob);
    setError(null);
  }, []);

  const convertVoiceToText = useCallback(async (audioBlob) => {
    try {
      const response = await apiService.voiceToText(audioBlob);
      return response.data.text;
    } catch (error) {
      console.error('Voice to text conversion failed:', error);
      throw new Error(t('errors.voice'));
    }
  }, [t]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      let description = formData.description;

      // Convert voice to text if using voice input
      if (inputMode === 'voice' && voiceBlob) {
        description = await convertVoiceToText(voiceBlob);
      }

      // Validate required fields
      if (!description || !formData.address) {
        throw new Error(t('errors.validation'));
      }

      const requestData = {
        description,
        address: formData.address,
        category: formData.category,
        language: formData.language,
      };

      // Generate complaint using AI
      const response = await apiService.generateComplaint(requestData);
      
      // Submit the complaint
      await apiService.submitComplaint({
        ...requestData,
        ...response.data,
      });

      // Reset form
      setFormData({
        description: '',
        address: '',
        category: 'other',
        language: 'en',
      });
      setVoiceBlob(null);
      
      // Call success callback
      onSubmitSuccess(response.data);

    } catch (err) {
      setError(err.message || t('errors.generic'));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          {t('form.title')}
        </h2>
        <p className="text-gray-600">
          {t('form.description')}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Input Mode Toggle */}
        <div className="flex space-x-4 p-1 bg-gray-100 rounded-lg">
          <button
            type="button"
            onClick={() => setInputMode('text')}
            className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-md transition-colors ${
              inputMode === 'text'
                ? 'bg-white text-primary-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <Type size={20} />
            <span>{t('form.textInput')}</span>
          </button>
          <button
            type="button"
            onClick={() => setInputMode('voice')}
            className={`flex-1 flex items-center justify-center space-x-2 py-2 px-4 rounded-md transition-colors ${
              inputMode === 'voice'
                ? 'bg-white text-primary-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-800'
            }`}
          >
            <Mic size={20} />
            <span>{t('form.voiceInput')}</span>
          </button>
        </div>

        {/* Description Input */}
        <div>
          <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
            {t('form.description')}
          </label>
          {inputMode === 'text' ? (
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              placeholder={t('form.descriptionPlaceholder')}
              className="form-textarea h-32"
              required
            />
          ) : (
            <VoiceRecorder
              onRecordingComplete={handleVoiceRecording}
              disabled={isSubmitting}
            />
          )}
        </div>

        {/* Address Input */}
        <div>
          <label htmlFor="address" className="block text-sm font-medium text-gray-700 mb-2">
            {t('form.address')}
          </label>
          <input
            type="text"
            id="address"
            name="address"
            value={formData.address}
            onChange={handleInputChange}
            placeholder={t('form.addressPlaceholder')}
            className="form-input"
            required
          />
        </div>

        {/* Category Selection */}
        <div>
          <label htmlFor="category" className="block text-sm font-medium text-gray-700 mb-2">
            {t('form.category')}
          </label>
          <select
            id="category"
            name="category"
            value={formData.category}
            onChange={handleInputChange}
            className="form-input"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {t(`form.categories.${category}`)}
              </option>
            ))}
          </select>
        </div>

        {/* Language Selection */}
        <div>
          <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-2">
            Translation Language
          </label>
          <select
            id="language"
            name="language"
            value={formData.language}
            onChange={handleInputChange}
            className="form-input"
          >
            {languages.map(lang => (
              <option key={lang.code} value={lang.code}>
                {lang.name}
              </option>
            ))}
          </select>
        </div>

        {/* Error Message */}
        {error && (
          <div className="flex items-center space-x-2 p-4 bg-red-50 border border-red-200 rounded-md">
            <AlertCircle className="text-red-500" size={20} />
            <span className="text-red-700">{error}</span>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting || (inputMode === 'voice' && !voiceBlob)}
          className="w-full btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSubmitting ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          ) : (
            <Send size={20} />
          )}
          <span>
            {isSubmitting ? t('form.submitting') : t('form.submit')}
          </span>
        </button>
      </form>
    </div>
  );
};

export default ComplaintForm;