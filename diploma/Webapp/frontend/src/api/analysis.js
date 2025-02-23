import axios from "axios";

export const sendMedicalData = async (formData, file) => {
  const data = new FormData();
  Object.keys(formData).forEach((key) => {
    data.append(key, formData[key]);
  });

  if (file) {
    data.append("file", file);
  }

  const response = await axios.post("http://localhost:5000/api/analysis", data, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};
