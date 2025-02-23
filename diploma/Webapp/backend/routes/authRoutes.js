const express = require("express");
const path = require("path");

// Import functions correctly
const { register, login, checkAuth } = require("../controllers/authController");
const authMiddleware = require("../middleware/authMiddleware");

const router = express.Router();

router.post("/register", register);
router.post("/login", login);
router.get("/check-auth", authMiddleware, checkAuth);

router.put("/update", authMiddleware, async (req, res) => {
  try {
    const userId = req.user.id;
    const updatedData = req.body;
    
    // Ensure correct model import
    const User = require("../models/User");

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { $set: updatedData },
      { new: true }
    );

    if (!updatedUser) {
      return res.status(404).json({ message: "User not found" });
    }

    res.json({ user: updatedUser, message: "Profile updated successfully" });
  } catch (error) {
    console.error("Error updating profile:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

module.exports = router;
