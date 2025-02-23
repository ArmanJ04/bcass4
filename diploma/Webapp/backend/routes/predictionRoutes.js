const express = require("express");
const router = express.Router();
const Prediction = require("../models/Prediction");

// Save Prediction
router.post("/save", async (req, res) => {
  try {
    const { email, prediction } = req.body;
    console.log("Received prediction data:", email, prediction); // ✅ Debugging log

    if (!email || prediction === undefined) {
      console.error("Missing email or prediction");
      return res.status(400).json({ error: "Email and prediction are required" });
    }

    const newPrediction = new Prediction({ email, prediction, timestamp: new Date() });
    await newPrediction.save();

    console.log("Prediction saved:", newPrediction); // ✅ Debugging log
    res.json({ message: "Prediction saved successfully" });
  } catch (error) {
    console.error("Error saving prediction:", error);
    res.status(500).json({ error: "Failed to save prediction" });
  }
});

// Fetch Prediction History
router.get("/history", async (req, res) => {
  try {
    const { email } = req.query;
    if (!email) return res.status(400).json({ error: "Email is required" });

    const history = await Prediction.find({ email }).sort({ timestamp: -1 });
    res.json({ history });
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch history" });
  }
});

module.exports = router;
