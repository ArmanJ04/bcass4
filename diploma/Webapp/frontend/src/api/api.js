export const predictCardioRisk = async (features) => {
    try {
      const response = await fetch("http://localhost:5000/api/ai/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ features }),
      });
  
      const data = await response.json();
      return data.prediction;
    } catch (error) {
      console.error("Error predicting:", error);
      return null;
    }
  };
  