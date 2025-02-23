const express = require("express");
const router = express.Router();
const Prediction = require("../models/Prediction");

// Получение истории предсказаний по email пользователя
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
