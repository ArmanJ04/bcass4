const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  age: { type: Number, required: true },
  gender: { type: String, required: true },
  height: { type: Number, required: false }, // ✅ Make sure it's included
  weight: { type: Number, required: false },
  smoking: { type: Boolean, default: false },
  alcohol: { type: Boolean, default: false },
  physicalActivity: { type: Number, required: false }, // ✅ This must be included
});

module.exports = mongoose.model("User", UserSchema);
