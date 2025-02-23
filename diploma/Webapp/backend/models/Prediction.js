const mongoose = require("mongoose");

const PredictionSchema = new mongoose.Schema({
  email: { type: String, required: true },
  prediction: { type: Number, required: true },
  timestamp: { type: Date, default: Date.now }, // ✅ Ensuring it's a valid Date
});

module.exports = mongoose.model("Prediction", PredictionSchema);
