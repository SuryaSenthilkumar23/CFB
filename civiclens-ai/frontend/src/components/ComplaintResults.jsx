import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Copy, Download, RotateCcw, Check, MapPin, FileText, Globe } from 'lucide-react';
import jsPDF from 'jspdf';

const ComplaintResults = ({ results, onNewReport }) => {
  const { t } = useTranslation();
  const [copiedField, setCopiedField] = useState(null);

  const handleCopy = async (text, fieldName) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(fieldName);
      setTimeout(() => setCopiedField(null), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const generatePDF = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.width;
    const margin = 20;
    const lineHeight = 10;
    let yPosition = margin;

    // Header
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('Civic Issue Complaint Report', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += lineHeight * 2;

    // Date
    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');
    doc.text(`Generated on: ${new Date().toLocaleDateString()}`, margin, yPosition);
    yPosition += lineHeight * 2;

    // Original Description
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Original Description:', margin, yPosition);
    yPosition += lineHeight;
    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');
    const originalLines = doc.splitTextToSize(results.originalText || '', pageWidth - 2 * margin);
    doc.text(originalLines, margin, yPosition);
    yPosition += originalLines.length * lineHeight + lineHeight;

    // Formal Complaint
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Formal Complaint:', margin, yPosition);
    yPosition += lineHeight;
    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');
    const complaintLines = doc.splitTextToSize(results.complaint_text || '', pageWidth - 2 * margin);
    doc.text(complaintLines, margin, yPosition);
    yPosition += complaintLines.length * lineHeight + lineHeight;

    // Check if we need a new page
    if (yPosition > doc.internal.pageSize.height - 40) {
      doc.addPage();
      yPosition = margin;
    }

    // Translated Text
    if (results.translated_text) {
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text('Translated Text:', margin, yPosition);
      yPosition += lineHeight;
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
      const translatedLines = doc.splitTextToSize(results.translated_text, pageWidth - 2 * margin);
      doc.text(translatedLines, margin, yPosition);
      yPosition += translatedLines.length * lineHeight + lineHeight;
    }

    // Summary
    if (results.summary) {
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text('Summary:', margin, yPosition);
      yPosition += lineHeight;
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
      const summaryLines = doc.splitTextToSize(results.summary, pageWidth - 2 * margin);
      doc.text(summaryLines, margin, yPosition);
      yPosition += summaryLines.length * lineHeight + lineHeight;
    }

    // Ward Information
    if (results.ward_guess) {
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text('Estimated Ward:', margin, yPosition);
      yPosition += lineHeight;
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
      doc.text(results.ward_guess, margin, yPosition);
    }

    // Save the PDF
    doc.save('civic-complaint-report.pdf');
  };

  const ResultCard = ({ title, content, icon: Icon, fieldName }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Icon className="text-primary-600" size={20} />
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        <button
          onClick={() => handleCopy(content, fieldName)}
          className="btn-secondary text-sm flex items-center space-x-1"
        >
          {copiedField === fieldName ? (
            <>
              <Check size={16} />
              <span>{t('results.actions.copied')}</span>
            </>
          ) : (
            <>
              <Copy size={16} />
              <span>{t('results.actions.copy')}</span>
            </>
          )}
        </button>
      </div>
      <div className="prose max-w-none">
        <p className="text-gray-700 leading-relaxed whitespace-pre-wrap">{content}</p>
      </div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {t('results.title')}
            </h2>
            <p className="text-gray-600">
              Generated on {new Date().toLocaleDateString()}
            </p>
          </div>
          <div className="flex space-x-3">
            <button
              onClick={generatePDF}
              className="btn-secondary flex items-center space-x-2"
            >
              <Download size={20} />
              <span>{t('results.actions.download')}</span>
            </button>
            <button
              onClick={onNewReport}
              className="btn-primary flex items-center space-x-2"
            >
              <RotateCcw size={20} />
              <span>{t('results.actions.newReport')}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Results Grid */}
      <div className="grid gap-6">
        {/* Original Description */}
        {results.originalText && (
          <ResultCard
            title={t('results.originalText')}
            content={results.originalText}
            icon={FileText}
            fieldName="original"
          />
        )}

        {/* Formal Complaint */}
        {results.complaint_text && (
          <ResultCard
            title={t('results.formalComplaint')}
            content={results.complaint_text}
            icon={FileText}
            fieldName="complaint"
          />
        )}

        {/* Translated Text */}
        {results.translated_text && (
          <ResultCard
            title={t('results.translatedText')}
            content={results.translated_text}
            icon={Globe}
            fieldName="translated"
          />
        )}

        {/* Summary */}
        {results.summary && (
          <ResultCard
            title={t('results.summary')}
            content={results.summary}
            icon={FileText}
            fieldName="summary"
          />
        )}

        {/* Ward Information */}
        {results.ward_guess && (
          <ResultCard
            title={t('results.wardGuess')}
            content={results.ward_guess}
            icon={MapPin}
            fieldName="ward"
          />
        )}
      </div>

      {/* Action Buttons */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-col sm:flex-row justify-center space-y-3 sm:space-y-0 sm:space-x-4">
          <button
            onClick={() => handleCopy(results.complaint_text, 'fullComplaint')}
            className="btn-secondary flex items-center justify-center space-x-2"
          >
            {copiedField === 'fullComplaint' ? (
              <>
                <Check size={20} />
                <span>{t('results.actions.copied')}</span>
              </>
            ) : (
              <>
                <Copy size={20} />
                <span>Copy Full Complaint</span>
              </>
            )}
          </button>
          <button
            onClick={generatePDF}
            className="btn-primary flex items-center justify-center space-x-2"
          >
            <Download size={20} />
            <span>{t('results.actions.download')}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default ComplaintResults;