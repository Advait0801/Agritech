import { useState, useEffect } from "react";
//import axios from "axios";
import { predict } from "../../API/YieldPredictApi";
import { auth } from "../../firebase"; // Ensure auth is correctly imported

function Yieldprediction() {
  const [formData, setFormData] = useState({
    Soil_Temp: "", N: "", P: "", K: "", Moisture: "", Humidity: "", Air_Temp: ""
  });

  const [prediction, setPrediction] = useState(null);
  const [user, setUser] = useState(null);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (user) {
        setUser(user); // Store user data if logged in
      } else {
        setUser(null); // Reset user data if logged out
      }
    });
    return unsubscribe; // Clean up subscription on unmount
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!user) {
      console.error("User is not logged in.");
      return;
    }

    const numericFormData = {
      Soil_Temp: Number(formData.Soil_Temp),
      N: Number(formData.N),
      P: Number(formData.P),
      K: Number(formData.K),
      Moisture: Number(formData.Moisture),
      Humidity: Number(formData.Humidity),
      Air_Temp: Number(formData.Air_Temp),
    };

    try {
      const token = await user.getIdToken();
      const res = await predict(numericFormData, token);
      setPrediction(res.data.yield);
    } catch (err) {
      console.error("Error:", err);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-gray-100 py-12 px-4">
      <h2 className="text-4xl font-semibold text-center text-gray-800 mb-8">Crop Yield Prediction</h2>
      <form onSubmit={handleSubmit} className="bg-white p-8 shadow-lg rounded-lg w-full max-w-lg">
        <div className="space-y-4">
          <input
            type="number"
            name="Soil_Temp"
            placeholder="Soil Temperature (°C)"
            value={formData.Soil_Temp}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="N"
            placeholder="Nitrogen (N)"
            value={formData.N}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="P"
            placeholder="Phosphorus (P)"
            value={formData.P}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="K"
            placeholder="Potassium (K)"
            value={formData.K}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="Moisture"
            placeholder="Soil Moisture (%)"
            value={formData.Moisture}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="Humidity"
            placeholder="Air Humidity (%)"
            value={formData.Humidity}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="Air_Temp"
            placeholder="Air Temperature (°C)"
            value={formData.Air_Temp}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
        </div>
        <button
          type="submit"
          className="w-full mt-6 bg-primary text-white py-3 rounded-md hover:bg-secondary focus:outline-none"
        >
          Predict
        </button>
      </form>
      {prediction !== null && (
        <div className="mt-6 bg-white p-4 shadow-lg rounded-md w-full max-w-lg">
          <h3 className="text-xl font-semibold text-center text-gray-800">
            Predicted Yield: <span className="text-primary">{prediction}</span> kgs/acre
          </h3>
        </div>
      )}
    </div>
  );
}

export default Yieldprediction;
