require("dotenv").config();
const mongoose = require("mongoose");

const MONGO_URI = process.env.MONGO_URI;

async function testMongoConnection() {
  try {
    await mongoose.connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true });
    console.log("✅ MongoDB connection successful!");
    mongoose.connection.close();
  } catch (error) {
    console.error("❌ MongoDB connection failed:", error);
  }
}

testMongoConnection();
