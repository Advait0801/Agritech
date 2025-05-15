import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { auth } from "../../firebase"; // Ensure auth is correctly imported
import disease from "../../Images/disease2.jpg";
import recommend from "../../Images/recommend.jpg";
import pred2 from "../../Images/pred2.jpg";
import insights from "../../Images/insights.webp";
import { XCircleIcon } from "@heroicons/react/24/solid";
import { onAuthStateChanged } from "firebase/auth"; // Import firebase auth function

const services = [
  {
    id: 1,
    title: "Crop Prediction",
    description: "Get crop yield based on the nutrient content of the soil.",
    image: pred2,
    link: "/Yieldprediction",
  },
  {
    id: 2,
    title: "Crop Recommendation",
    description: "Get to know which crops are compatible according the nutrient content of soil.",
    image: recommend,
    link: "/CropRecommendation",
  },
  {
    id: 3,
    title: "Disease Prediction",
    description: "Identify the disease caught by the crops and get solutions accordingly.",
    image: disease,
    link: "/DiseasePrediction",
  },
  {
    id: 4,
    title: "Visual Insights",
    description: "Runtime visuals of the data collected from the sensors.",
    image: insights,
    link: "/Visualization",
  },
];

const Services = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null); // State to track the user
  const [showMessage, setShowMessage] = useState(false); // State for alert message

  useEffect(() => {
    // Listen to authentication state changes
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    
    // Cleanup the subscription when the component unmounts
    return () => unsubscribe();
  }, []);

  const handleClick = (event, link) => {
    event.preventDefault();
    if (user) {
      navigate(link); // If user is authenticated, navigate to the link
    } else {
      setShowMessage(true); // Show alert if not authenticated
      setTimeout(() => setShowMessage(false), 4000); // Hide after 4 seconds
    }
  };

  return (
    <div className="services-container bg-gray-100 py-16 px-4">
      <h2 className="text-4xl font-bold text-center text-black mb-8">
        Our Services
      </h2>

      {/* Enhanced Alert Message */}
      {showMessage && (
        <div className="flex items-center justify-between bg-red-100 text-red-800 border border-red-300 px-4 py-3 rounded-md shadow-md max-w-md mx-auto mb-6">
          <div className="flex items-center">
            <XCircleIcon className="h-5 w-5 mr-2 text-red-500" />
            <span>You need to log in to access this feature.</span>
          </div>
          <button onClick={() => setShowMessage(false)} className="text-red-500 hover:text-red-700">
            ✖
          </button>
        </div>
      )}

      <div className="max-w-6xl mx-auto grid gap-8 md:grid-cols-3 sm:grid-cols-1">
        {services.map((service) => (
          <div key={service.id} className="bg-white shadow-lg rounded-lg p-6">
            <img
              src={service.image}
              alt={service.title}
              className="w-full h-48 object-cover rounded-md"
            />
            <h3 className="text-2xl font-semibold mt-4">{service.title}</h3>
            <p className="text-gray-700 mt-2">{service.description}</p>
            <a
              href={service.link}
              onClick={(event) => handleClick(event, service.link)}
              className="mt-4 block bg-primary text-white py-2 rounded-md text-center hover:bg-secondary"
            >
              Explore
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Services;
