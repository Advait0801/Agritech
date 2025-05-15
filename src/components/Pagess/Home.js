import React from "react";
import { Link } from "react-router-dom";
//import { GiFarmTractor } from "react-icons/gi";
import { GiFarmer } from "react-icons/gi";

import { FaLeaf, FaSeedling, FaSolarPanel, FaSearch } from "react-icons/fa";
import farmImage from "../../Images/cover.JPG";
import solar from "../../Images/solar.jpg" ; 
import soil_analysis from "../../Images/soil_analysis.jpg" ; 
//import analysisImage from "../../Images/analysis.jpg";
//import energyImage from "../../Images/clean-energy.jpg";

const Home = () => {
  return (
    <div className="bg-gray-100 min-h-screen">
      {/* Hero Section */}
      <section
        className="relative text-center text-white py-32 px-10 md:px-12"
        style={{ background: `url(${farmImage}) center/cover no-repeat` }}
      >
        <div className="bg-black bg-opacity-50 p-8 rounded-lg inline-block">
          <GiFarmer className="text-primary text-7xl mx-auto mb-4" />
          <h1 className="text-5xl font-bold">Welcome to AgriTech</h1>
          <p className="text-lg mt-4 max-w-2xl mx-auto">
            Empowering farmers with cutting-edge technology to optimize agriculture.
          </p>
          <Link to="/services">
            <button className="mt-6 px-6 py-3 bg-primary text-white font-semibold text-lg rounded-md hover:bg-green-700 transition">
              Explore Services
            </button>
          </Link>
        </div>
      </section>

      {/* Why Choose AgriTech? */}
      <section className="py-16 px-6 md:px-12 bg-white">
        <h2 className="text-4xl font-bold text-center text-gray-800">Why Choose AgriTech?</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-10 mt-12">
          <div className="text-center bg-gray-50 p-6 rounded-lg shadow-lg">
            <FaLeaf className="text-green-500 text-6xl mx-auto" />
            <h3 className="text-xl font-semibold mt-4">Sustainable Farming</h3>
            <p className="mt-2">Implementing eco-friendly farming techniques.</p>
          </div>
          <div className="text-center bg-gray-50 p-6 rounded-lg shadow-lg">
            <FaSeedling className="text-yellow-500 text-6xl mx-auto" />
            <h3 className="text-xl font-semibold mt-4">Better Soil Health</h3>
            <p className="mt-2">In-depth soil analysis for future recommendations.</p>
          </div>
          <div className="text-center bg-gray-50 p-6 rounded-lg shadow-lg">
            <FaSolarPanel className="text-blue-500 text-6xl mx-auto" />
            <h3 className="text-xl font-semibold mt-4">Clean Energy</h3>
            <p className="mt-2">Using renewable energy to power agriculture.</p>
          </div>
          <div className="text-center bg-gray-50 p-6 rounded-lg shadow-lg">
            <FaSearch className="text-red-500 text-6xl mx-auto" />
            <h3 className="text-xl font-semibold mt-4">Smart Data Insights</h3>
            <p className="mt-2">AI-driven insights for improved decision-making.</p>
          </div>
        </div>
      </section>

      {/* Featured Image Section */}
      <section className="py-16 px-6 md:px-12 bg-gray-150">
        <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-10 items-center">
          <img src={soil_analysis} alt="Soil Analysis" className="w-full rounded-lg shadow-lg" />
          <div>
            <h3 className="text-3xl font-bold text-gray-800">Advanced Soil Analysis</h3>
            <p className="mt-4 text-gray-700">
              We provide AI-powered soil analysis for better crop yield and sustainability.
            </p>
          </div>
        </div>
      </section>

      <section className="py-16 px-6 md:px-12 bg-white">
        <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-10 items-center">
          <div>
            <h3 className="text-3xl font-bold text-gray-800">Renewable Energy Integration</h3>
            <p className="mt-4 text-gray-700">
              We have powered our technology with clean, renewable solar energy solutions.
            </p>
          </div>
          <img src={solar} alt="Clean Energy" className="w-full rounded-lg shadow-lg" />
        </div>
      </section>
    </div>
  );
};

export default Home;
