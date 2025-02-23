const jwt = require("jsonwebtoken");

const SECRET_KEY = process.env.JWT_SECRET

const authMiddleware = (req, res, next) => {
  const token = req.cookies.token;
  if (!token) return res.status(401).json({ message: "Unauthorized" });

  jwt.verify(token, SECRET_KEY, (err, decoded) => {
    if (err) return res.status(403).json({ message: "Invalid token" });
    
    req.user = decoded; // Attach user data to request
    next();
  });
};

module.exports = authMiddleware;
