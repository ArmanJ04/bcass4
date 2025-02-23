import React, { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { useNavigate } from "react-router-dom";

function Profile() {
  const { user, updateUser } = useAuth();
  const navigate = useNavigate();
  const [editable, setEditable] = useState(false);
  const [history, setHistory] = useState([]);
  
  // ✅ Read user from localStorage if available
  const [formData, setFormData] = useState(() => {
    const storedUser = JSON.parse(localStorage.getItem("user"));
    return storedUser || {
      name: "",
      age: "",
      height: "",
      weight: "",
      gender: "",
      smoking: false,
      alcohol: false,
      physicalActivity: "",
    };
  });

  useEffect(() => {
    if (!user) {
      navigate("/login");
      return;
    }

    setFormData({ ...user });
    fetchHistory(user.email);
  }, [user, navigate]);

  const fetchHistory = async (email) => {
    try {
      const response = await fetch(`http://localhost:5000/api/prediction/history?email=${email}`);
      const data = await response.json();
      if (data.history) {
        setHistory(data.history);
      }
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSave = async () => {
    await updateUser(formData); // ✅ Correct function call
    setEditable(false);
    localStorage.setItem("user", JSON.stringify(formData)); // ✅ Save updated data in localStorage
  };

  if (!user) return <p>Loading...</p>;

  return (
    <div className="profile-container">
      <h1>Profile</h1>
      <div className="profile-info">
        {editable ? (
          <>
            <input type="text" name="name" value={formData.name} onChange={handleChange} placeholder="Name" />
            <input type="number" name="age" value={formData.age} onChange={handleChange} placeholder="Age" />
            <input type="number" name="height" value={formData.height} onChange={handleChange} placeholder="Height (cm)" />
            <input type="number" name="weight" value={formData.weight} onChange={handleChange} placeholder="Weight (kg)" />
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
            <label>
              <input type="checkbox" name="smoking" checked={formData.smoking} onChange={handleChange} /> Smoking
            </label>
            <label>
              <input type="checkbox" name="alcohol" checked={formData.alcohol} onChange={handleChange} /> Alcohol
            </label>
            <input type="number" name="physicalActivity" value={formData.physicalActivity} onChange={handleChange} placeholder="Hours per week" />
            <button onClick={handleSave}>Save</button>
          </>
        ) : (
          <>
            <p><strong>Name:</strong> {formData.name}</p>
            <p><strong>Age:</strong> {formData.age}</p>
            <p><strong>Height:</strong> {formData.height} cm</p>
            <p><strong>Weight:</strong> {formData.weight} kg</p>
            <p><strong>Gender:</strong> {formData.gender}</p>
            <p><strong>Smoking:</strong> {formData.smoking ? "Yes" : "No"}</p>
            <p><strong>Alcohol Intake:</strong> {formData.alcohol ? "Yes" : "No"}</p>
            <p><strong>Physical Activity:</strong> {formData.physicalActivity} hours per week</p>
            <button onClick={() => setEditable(true)}>Edit</button>
          </>
        )}
      </div>

      <div className="history-section">
        <h2>Prediction History</h2>
        {history.length > 0 ? (
          <ul>
            {history.map((item, index) => (
              <li key={index}>
                <p><strong>Prediction:</strong> {item.prediction}</p>
                <p><strong>Date:</strong> {new Date(item.timestamp).toLocaleString()}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p>No prediction history available.</p>
        )}
      </div>
    </div>
  );
}

export default Profile;
