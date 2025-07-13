const mongoose = require('mongoose');

const complaintSchema = new mongoose.Schema({
  // Original user input
  originalDescription: {
    type: String,
    required: true,
    trim: true,
  },
  
  // Address information
  address: {
    type: String,
    required: true,
    trim: true,
  },
  
  // Category of the complaint
  category: {
    type: String,
    required: true,
    enum: ['pothole', 'garbage', 'streetlight', 'water', 'drainage', 'other'],
    default: 'other',
  },
  
  // AI-generated formal complaint
  formalComplaint: {
    type: String,
    required: true,
  },
  
  // Translated text
  translatedText: {
    type: String,
  },
  
  // Translation language
  language: {
    type: String,
    required: true,
    enum: ['en', 'hi', 'ta'],
    default: 'en',
  },
  
  // AI-generated summary
  summary: {
    type: String,
  },
  
  // Ward estimation
  wardGuess: {
    type: String,
  },
  
  // Geolocation data
  coordinates: {
    type: {
      type: String,
      enum: ['Point'],
    },
    coordinates: {
      type: [Number], // [longitude, latitude]
    },
  },
  
  // Status tracking
  status: {
    type: String,
    enum: ['submitted', 'in_progress', 'resolved', 'closed'],
    default: 'submitted',
  },
  
  // Priority level
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium',
  },
  
  // Additional metadata
  metadata: {
    inputMode: {
      type: String,
      enum: ['text', 'voice'],
      default: 'text',
    },
    processingTime: {
      type: Number, // in milliseconds
    },
    aiModel: {
      type: String,
    },
    voiceToTextAccuracy: {
      type: Number,
    },
  },
  
  // Timestamps
  submittedAt: {
    type: Date,
    default: Date.now,
  },
  
  updatedAt: {
    type: Date,
    default: Date.now,
  },
  
  // User information (if authentication is added later)
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  
  // Contact information
  contactInfo: {
    name: String,
    email: String,
    phone: String,
  },
  
  // Resolution information
  resolution: {
    resolvedAt: Date,
    resolvedBy: String,
    resolution: String,
    feedback: String,
  },
}, {
  timestamps: true,
});

// Create indexes for better query performance
complaintSchema.index({ address: 'text', originalDescription: 'text' });
complaintSchema.index({ category: 1, status: 1 });
complaintSchema.index({ submittedAt: -1 });
complaintSchema.index({ coordinates: '2dsphere' });

// Virtual for complaint age
complaintSchema.virtual('age').get(function() {
  return Date.now() - this.submittedAt;
});

// Method to update status
complaintSchema.methods.updateStatus = function(newStatus, userId) {
  this.status = newStatus;
  this.updatedAt = new Date();
  
  if (newStatus === 'resolved') {
    this.resolution.resolvedAt = new Date();
    this.resolution.resolvedBy = userId;
  }
  
  return this.save();
};

// Static method to get complaints by category
complaintSchema.statics.getByCategory = function(category, limit = 10) {
  return this.find({ category })
    .sort({ submittedAt: -1 })
    .limit(limit);
};

// Static method to get complaints by ward
complaintSchema.statics.getByWard = function(ward, limit = 10) {
  return this.find({ wardGuess: new RegExp(ward, 'i') })
    .sort({ submittedAt: -1 })
    .limit(limit);
};

// Pre-save middleware to update timestamps
complaintSchema.pre('save', function(next) {
  if (this.isModified() && !this.isNew) {
    this.updatedAt = new Date();
  }
  next();
});

module.exports = mongoose.model('Complaint', complaintSchema);