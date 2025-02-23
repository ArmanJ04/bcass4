import React, { useState } from "react";

const MedicalInputForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    age: "",
    bloodPressure: "",
    cholesterol: "",
    heartRate: "",
  });
  const [file, setFile] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData, file);
  };

  return (
    <div>
      <h2>Medical Data Input</h2>
      <form onSubmit={handleSubmit}>
        <label>Age:</label>
        <input type="number" name="age" value={formData.age} onChange={handleChange} required />

        <label>Blood Pressure:</label>
        <input type="number" name="bloodPressure" value={formData.bloodPressure} onChange={handleChange} required />

        <label>Cholesterol:</label>
        <input type="number" name="cholesterol" value={formData.cholesterol} onChange={handleChange} required />

        <label>Heart Rate:</label>
        <input type="number" name="heartRate" value={formData.heartRate} onChange={handleChange} required />

        <label>Upload JSON File:</label>
        <input type="file" accept=".json" onChange={handleFileChange} />

        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

export default MedicalInputForm;
