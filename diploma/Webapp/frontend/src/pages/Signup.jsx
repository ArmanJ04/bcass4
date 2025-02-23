import { useState, useContext } from "react";
import { AuthContext } from "../context/AuthContext";
import { useNavigate, Link } from "react-router-dom";

function Signup() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: "",
    age: "",
    gender: "",
    height: "",
    weight: "",
    smoking: false,
    alcohol: false,
    physicalActivity: "",
  });

  const { signup } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.password.length < 8) {
      window.alert("Password must be at least 8 characters long.");
      return;
    }

    try {
      await signup(formData);
      window.alert("Signup successful! Redirecting to your profile...");
      navigate("/profile");
    } catch (error) {
      window.alert("Signup failed. Please try again.");
    }
  };

  return (
    <div className="container">
      <h2>Sign Up</h2>
      <form onSubmit={handleSubmit}>
        <input type="text" name="name" placeholder="Full Name" value={formData.name} onChange={handleChange} required />
        <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} required />
        <input type="password" name="password" placeholder="Password (min. 8 chars)" value={formData.password} onChange={handleChange} required />
        <input type="number" name="age" placeholder="Age" value={formData.age} onChange={handleChange} required />
        <select name="gender" value={formData.gender} onChange={handleChange} required>
          <option value="">Select Gender</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
        <input type="number" name="height" placeholder="Height (cm)" value={formData.height} onChange={handleChange} required />
        <input type="number" name="weight" placeholder="Weight (kg)" value={formData.weight} onChange={handleChange} required />
        <label>
          <input type="checkbox" name="smoking" checked={formData.smoking} onChange={handleChange} />
          Smoking
        </label>
        <label>
          <input type="checkbox" name="alcohol" checked={formData.alcohol} onChange={handleChange} />
          Alcohol Intake
        </label>
        <input type="text" name="physicalActivity" placeholder="Physical Activity (hours per week)" value={formData.physicalActivity} onChange={handleChange} required />
        <button type="submit">Sign Up</button>
      </form>
      <p>Already have an account? <Link to="/login">Login here</Link></p>
    </div>
  );
}

export default Signup;
