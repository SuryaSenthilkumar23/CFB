import React, { useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Mic, MicOff, Play, Pause, Square, RotateCcw } from 'lucide-react';
import { useVoiceRecording } from '../hooks/useVoiceRecording';

const VoiceRecorder = ({ onRecordingComplete, disabled = false }) => {
  const { t } = useTranslation();
  const {
    isRecording,
    isPaused,
    recordedBlob,
    error,
    formattedTime,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    resetRecording,
    cleanup,
  } = useVoiceRecording();

  useEffect(() => {
    if (recordedBlob) {
      onRecordingComplete(recordedBlob);
    }
  }, [recordedBlob, onRecordingComplete]);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  const handleStartRecording = async () => {
    try {
      await startRecording();
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };

  const RecordingButton = () => {
    if (isRecording) {
      return (
        <div className="flex items-center space-x-2">
          <button
            onClick={isPaused ? resumeRecording : pauseRecording}
            className="p-3 bg-yellow-500 text-white rounded-full hover:bg-yellow-600 transition-colors"
            disabled={disabled}
          >
            {isPaused ? <Play size={20} /> : <Pause size={20} />}
          </button>
          <button
            onClick={stopRecording}
            className="p-3 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
            disabled={disabled}
          >
            <Square size={20} />
          </button>
        </div>
      );
    }

    return (
      <button
        onClick={handleStartRecording}
        className="p-4 bg-primary-600 text-white rounded-full hover:bg-primary-700 transition-colors shadow-lg"
        disabled={disabled}
      >
        <Mic size={24} />
      </button>
    );
  };

  const RecordingStatus = () => {
    if (isRecording) {
      return (
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-medium text-red-600">
            {isPaused ? t('common.paused') : t('form.recording')}
          </span>
          <span className="text-sm text-gray-500">{formattedTime}</span>
        </div>
      );
    }

    if (recordedBlob) {
      return (
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-sm font-medium text-green-600">
            {t('common.recorded')}
          </span>
          <button
            onClick={resetRecording}
            className="p-1 text-gray-500 hover:text-gray-700 transition-colors"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex flex-col items-center space-y-4">
        <div className="flex items-center space-x-4">
          <RecordingButton />
          <div className="flex flex-col items-center">
            <RecordingStatus />
            {error && (
              <div className="text-red-500 text-sm mt-2">
                {error}
              </div>
            )}
          </div>
        </div>
        
        <div className="text-center">
          <p className="text-sm text-gray-600">
            {isRecording
              ? t('form.recording')
              : recordedBlob
              ? t('common.recorded')
              : t('form.startRecording')
            }
          </p>
        </div>
        
        {recordedBlob && (
          <div className="w-full">
            <audio
              controls
              src={URL.createObjectURL(recordedBlob)}
              className="w-full"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceRecorder;