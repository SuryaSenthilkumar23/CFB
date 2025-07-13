const express = require('express');
const multer = require('multer');
const { body, validationResult } = require('express-validator');
const axios = require('axios');
const Complaint = require('../models/Complaint');
const logger = require('../utils/logger');

const router = express.Router();

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept audio files
    if (file.mimetype.startsWith('audio/')) {
      cb(null, true);
    } else {
      cb(new Error('Only audio files are allowed'), false);
    }
  },
});

// Validation middleware
const validateComplaint = [
  body('description').trim().notEmpty().withMessage('Description is required'),
  body('address').trim().notEmpty().withMessage('Address is required'),
  body('category').isIn(['pothole', 'garbage', 'streetlight', 'water', 'drainage', 'other']).withMessage('Invalid category'),
  body('language').isIn(['en', 'hi', 'ta']).withMessage('Invalid language'),
];

// AI Microservice URL
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

// Helper function to call AI microservice
const callAIService = async (endpoint, data) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}${endpoint}`, data, {
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return response.data;
  } catch (error) {
    logger.error(`AI Service call failed for ${endpoint}:`, error.message);
    throw new Error(`AI processing failed: ${error.message}`);
  }
};

// @route POST /api/voice-to-text
// @desc Convert voice to text
// @access Public
router.post('/voice-to-text', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Audio file is required' });
    }

    const formData = new FormData();
    formData.append('audio', req.file.buffer, {
      filename: 'audio.webm',
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${AI_SERVICE_URL}/voice-to-text`, formData, {
      timeout: 30000,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    res.json({
      text: response.data.text,
      confidence: response.data.confidence,
    });

  } catch (error) {
    logger.error('Voice to text conversion failed:', error);
    res.status(500).json({ error: 'Voice to text conversion failed' });
  }
});

// @route POST /api/generate-complaint
// @desc Generate formal complaint using AI
// @access Public
router.post('/generate-complaint', validateComplaint, async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    const { description, address, category, language } = req.body;
    const startTime = Date.now();

    // Call AI microservice to generate complaint
    const aiResponse = await callAIService('/generate-complaint', {
      description,
      address,
      category,
      language,
    });

    const processingTime = Date.now() - startTime;

    res.json({
      complaint_text: aiResponse.complaint_text,
      translated_text: aiResponse.translated_text,
      summary: aiResponse.summary,
      ward_guess: aiResponse.ward_guess,
      originalText: description,
      processingTime,
    });

  } catch (error) {
    logger.error('Complaint generation failed:', error);
    res.status(500).json({ error: 'Failed to generate complaint' });
  }
});

// @route POST /api/translate
// @desc Translate text to specified language
// @access Public
router.post('/translate', async (req, res) => {
  try {
    const { text, targetLanguage } = req.body;

    if (!text || !targetLanguage) {
      return res.status(400).json({ error: 'Text and target language are required' });
    }

    const aiResponse = await callAIService('/translate', {
      text,
      target_language: targetLanguage,
    });

    res.json({
      translatedText: aiResponse.translated_text,
      sourceLanguage: aiResponse.source_language,
      targetLanguage: aiResponse.target_language,
    });

  } catch (error) {
    logger.error('Translation failed:', error);
    res.status(500).json({ error: 'Translation failed' });
  }
});

// @route POST /api/submit
// @desc Submit a complaint to the database
// @access Public
router.post('/submit', async (req, res) => {
  try {
    const {
      description,
      address,
      category,
      language,
      complaint_text,
      translated_text,
      summary,
      ward_guess,
      originalText,
      processingTime,
      inputMode,
    } = req.body;

    // Create new complaint
    const complaint = new Complaint({
      originalDescription: originalText || description,
      address,
      category,
      language,
      formalComplaint: complaint_text,
      translatedText: translated_text,
      summary,
      wardGuess: ward_guess,
      metadata: {
        inputMode: inputMode || 'text',
        processingTime,
        aiModel: 'openrouter-mixtral',
      },
    });

    // Save to database
    await complaint.save();

    logger.info(`New complaint submitted: ${complaint._id}`);

    res.status(201).json({
      message: 'Complaint submitted successfully',
      complaintId: complaint._id,
      submittedAt: complaint.submittedAt,
    });

  } catch (error) {
    logger.error('Complaint submission failed:', error);
    res.status(500).json({ error: 'Failed to submit complaint' });
  }
});

// @route GET /api/complaints
// @desc Get all complaints (with pagination)
// @access Public
router.get('/complaints', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const { category, status, ward } = req.query;

    // Build query
    const query = {};
    if (category) query.category = category;
    if (status) query.status = status;
    if (ward) query.wardGuess = new RegExp(ward, 'i');

    const complaints = await Complaint.find(query)
      .sort({ submittedAt: -1 })
      .skip(skip)
      .limit(limit)
      .select('-__v');

    const total = await Complaint.countDocuments(query);

    res.json({
      complaints,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit),
      },
    });

  } catch (error) {
    logger.error('Failed to fetch complaints:', error);
    res.status(500).json({ error: 'Failed to fetch complaints' });
  }
});

// @route GET /api/complaints/:id
// @desc Get a specific complaint by ID
// @access Public
router.get('/complaints/:id', async (req, res) => {
  try {
    const complaint = await Complaint.findById(req.params.id);

    if (!complaint) {
      return res.status(404).json({ error: 'Complaint not found' });
    }

    res.json(complaint);

  } catch (error) {
    logger.error('Failed to fetch complaint:', error);
    res.status(500).json({ error: 'Failed to fetch complaint' });
  }
});

// @route PUT /api/complaints/:id/status
// @desc Update complaint status
// @access Public
router.put('/complaints/:id/status', async (req, res) => {
  try {
    const { status } = req.body;

    if (!['submitted', 'in_progress', 'resolved', 'closed'].includes(status)) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    const complaint = await Complaint.findById(req.params.id);

    if (!complaint) {
      return res.status(404).json({ error: 'Complaint not found' });
    }

    await complaint.updateStatus(status);

    res.json({
      message: 'Status updated successfully',
      complaint,
    });

  } catch (error) {
    logger.error('Failed to update complaint status:', error);
    res.status(500).json({ error: 'Failed to update complaint status' });
  }
});

// @route GET /api/complaints/stats
// @desc Get complaint statistics
// @access Public
router.get('/complaints/stats', async (req, res) => {
  try {
    const stats = await Complaint.aggregate([
      {
        $group: {
          _id: null,
          totalComplaints: { $sum: 1 },
          byCategory: {
            $push: {
              category: '$category',
              count: 1,
            },
          },
          byStatus: {
            $push: {
              status: '$status',
              count: 1,
            },
          },
        },
      },
    ]);

    res.json(stats[0] || { totalComplaints: 0, byCategory: [], byStatus: [] });

  } catch (error) {
    logger.error('Failed to fetch complaint stats:', error);
    res.status(500).json({ error: 'Failed to fetch complaint stats' });
  }
});

module.exports = router;