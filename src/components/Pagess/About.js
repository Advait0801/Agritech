import React from "react";
import { FaGlobe, FaUsers, FaLightbulb, FaEnvelope } from "react-icons/fa";
import { motion } from "framer-motion";
import about1 from "../../Images/about1.JPG";
import about2 from "../../Images/about2.JPG";
import about3 from "../../Images/about3.JPG";
import about4 from "../../Images/about4.JPG";
import about5 from "../../Images/about5.JPG";
import about6 from "../../Images/about6.JPG";


const AboutUs = () => {
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center py-12 px-6">
      {/* Header Section */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="text-center max-w-4xl"
      >
        <h1 className="text-5xl font-extrabold text-primary">About Us</h1>
        <p className="text-lg text-gray-600 mt-4">
          Empowering agriculture with AI-driven insights and Machine Learning models. We help farmers make informed decisions for better yield and sustainability.
        </p>
      </motion.div>

      {/* Mission & Vision Section */}
      <div className="grid md:grid-cols-2 gap-8 mt-12 max-w-5xl">
        {[{ icon: FaLightbulb, title: "Our Mission", text: "To revolutionize modern farming through AI and data-driven technology, ensuring efficient and sustainable agriculture." },
          { icon: FaGlobe, title: "Our Vision", text: "A world where every farmer has access to intelligent tools for maximizing productivity and profitability." }
        ].map((item, index) => (
          <motion.div 
            key={index}
            initial={{ opacity: 0, y: 20 }} 
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.2 }}
            className="bg-white p-6 rounded-lg shadow-lg hover:shadow-2xl transform transition-all duration-300 hover:scale-105 text-center"
          >
            <item.icon className="text-primary text-4xl mx-auto" />
            <h2 className="text-2xl font-bold mt-4">{item.title}</h2>
            <p className="text-gray-700 mt-2">{item.text}</p>
          </motion.div>
        ))}
      </div>

      {/* Image Gallery */}
      <div className="mt-16 max-w-5xl w-full">
        <h2 className="text-3xl font-bold text-center text-primary">Our Work in Action</h2>
        <motion.div 
          initial={{ opacity: 0 }} 
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 mt-8"
        >
          {[about1,about2,about3,
            about4,about5,about6
          ].map((src, index) => (
            <motion.div key={index} whileHover={{ scale: 1.1 }} className="overflow-hidden rounded-xl shadow-lg">
              <img src={src} alt="Farming Activity" className="w-full h-56 object-cover transition-transform duration-300 hover:scale-110" />
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Contact Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.5 }}
        className="mt-12 bg-white p-6 rounded-lg shadow-lg max-w-4xl w-full text-center hover:shadow-2xl transform transition-all duration-300 hover:scale-105"
      >
        <h2 className="text-3xl font-bold text-primary">Get in Touch</h2>
        <p className="text-gray-700 mt-2">Have questions? Want to collaborate? Reach out to us!</p>
        <p className="text-lg font-semibold text-primary mt-2 flex justify-center items-center gap-2">
          <FaEnvelope /> ad.naik@gmail.com
        </p>
      </motion.div>
    </div>
  );
};

export default AboutUs;
