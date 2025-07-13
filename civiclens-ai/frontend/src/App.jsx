import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, ArrowLeft } from 'lucide-react';
import ComplaintForm from './components/ComplaintForm';
import ComplaintResults from './components/ComplaintResults';
import LanguageSwitcher from './components/LanguageSwitcher';

function App() {
  const { t } = useTranslation();
  const [currentView, setCurrentView] = useState('form'); // 'form' or 'results'
  const [complaintResults, setComplaintResults] = useState(null);

  const handleSubmitSuccess = (results) => {
    setComplaintResults(results);
    setCurrentView('results');
  };

  const handleNewReport = () => {
    setComplaintResults(null);
    setCurrentView('form');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              {currentView === 'results' && (
                <button
                  onClick={handleNewReport}
                  className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
                >
                  <ArrowLeft size={20} />
                  <span>Back to Form</span>
                </button>
              )}
              <div className="flex items-center space-x-3">
                <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
                  <AlertTriangle className="text-white" size={24} />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">
                    {t('app.title')}
                  </h1>
                  <p className="text-sm text-gray-600">
                    {t('app.subtitle')}
                  </p>
                </div>
              </div>
            </div>
            <LanguageSwitcher />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentView === 'form' && (
          <div className="space-y-8">
            {/* Hero Section */}
            <div className="text-center">
              <h2 className="text-3xl font-bold text-gray-900 sm:text-4xl">
                {t('app.title')}
              </h2>
              <p className="mt-4 text-lg text-gray-600 max-w-2xl mx-auto">
                {t('app.description')}
              </p>
            </div>

            {/* Form */}
            <ComplaintForm onSubmitSuccess={handleSubmitSuccess} />

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mx-auto mb-4">
                  <AlertTriangle className="text-primary-600" size={24} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Voice & Text Input
                </h3>
                <p className="text-gray-600">
                  Report issues using your voice or by typing. Our AI will process both inputs seamlessly.
                </p>
              </div>
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mx-auto mb-4">
                  <AlertTriangle className="text-primary-600" size={24} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  AI-Powered Processing
                </h3>
                <p className="text-gray-600">
                  Our AI generates formal complaints, provides translations, and estimates ward information.
                </p>
              </div>
              <div className="bg-white rounded-lg shadow-md p-6 text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mx-auto mb-4">
                  <AlertTriangle className="text-primary-600" size={24} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Multi-language Support
                </h3>
                <p className="text-gray-600">
                  Get your complaints translated to regional languages like Hindi and Tamil.
                </p>
              </div>
            </div>
          </div>
        )}

        {currentView === 'results' && complaintResults && (
          <ComplaintResults
            results={complaintResults}
            onNewReport={handleNewReport}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p>
              © 2024 CivicLens AI. Powered by artificial intelligence for better civic engagement.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;