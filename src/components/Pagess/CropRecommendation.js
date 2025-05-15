import { useState, useEffect } from "react";
import { auth } from "../../firebase"; // Ensure auth is correctly imported
import { FaSeedling, FaAppleAlt, FaCarrot } from "react-icons/fa"; // You can replace these with actual crop icons

function CropRecommendation() {
  const [formData, setFormData] = useState({
    N: "",
    P: "",
    K: "",
    Temp: "",
    Humidity: "",
  });

  const [recommendation, setRecommendation] = useState(null);
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

    // Check if user is logged in
    if (!user) {
      console.error("User is not logged in.");
      return;
    }

    const numericFormData = {
      N: Number(formData.N),
      P: Number(formData.P),
      K: Number(formData.K),
      temperature: Number(formData.Temp),
      humidity: Number(formData.Humidity),
    };

    try {
      const token = await user.getIdToken();

      const response = await fetch("http://localhost:5001/predict_crop", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
        body: JSON.stringify(numericFormData),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const result = await response.json();
      setRecommendation(result.predictions); // Assuming the result contains the crop recommendations
    } catch (err) {
      console.error("Error:", err);
    }
  };

  const renderCropIcon = (crop) => {
    // Add custom icons for specific crops
    switch (crop) {
      case "Wheat":
        return <FaSeedling className="text-2xl text-green-500" />;
      case "Rice":
        return <FaAppleAlt className="text-2xl text-red-500" />;
      case "Maize":
        return <FaCarrot className="text-2xl text-orange-500" />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-center items-center bg-gray-100 py-12 px-4">
      <h2 className="text-4xl font-semibold text-center text-gray-800 mb-8">Crop Recommendation</h2>
      <form onSubmit={handleSubmit} className="bg-white p-8 shadow-lg rounded-lg w-full max-w-lg">
        <div className="space-y-4">
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
            name="Temp"
            placeholder="Temperature (°C)"
            value={formData.Temp}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
          <input
            type="number"
            name="Humidity"
            placeholder="Humidity (%)"
            value={formData.Humidity}
            onChange={handleChange}
            required
            className="w-full p-3 border border-gray-300 rounded-md"
          />
        </div>
        <button
          type="submit"
          className="w-full mt-6 bg-primary text-white py-3 rounded-md hover:bg-secondary focus:outline-none"
        >
          Get Recommendation
        </button>
      </form>

      {recommendation && (
        <div className="mt-6 bg-white p-4 shadow-lg rounded-md w-full max-w-lg">
          <h3 className="text-xl font-semibold text-center text-gray-800">Recommendation:</h3>
          {recommendation.map((item, index) => (
            <div key={index} className="flex items-center space-x-4 text-center mb-3">
              {renderCropIcon(item.crop)} {/* Render crop icon */}
              <p className="text-lg font-semibold">{item.crop}: {item.probability.toFixed(2)}%</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default CropRecommendation;
