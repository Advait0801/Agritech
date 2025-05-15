import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Home from "./components/Pagess/Home"
import About from "./components/Pagess/About";
import Services from "./components/Pagess/Services";
import Contact from "./components/Pagess/Contact";
import Footer from "./components/Footer/Footer";
import Login from "./components/Pagess/Login";
import Signup from "./components/Pagess/Signup";
import Yieldprediction from "./components/Pagess/Yieldprediction";
import CropRecommendation from "./components/Pagess/CropRecommendation";
import Visualization from "./components/Pagess/Visualization";
import Profile from "./components/Pagess/Profile";
import DiseasePrediction from "./components/Pagess/DiseasePrediction";


const App = () => {
  return (
    <Router>
      <div className="overflow-x-hidden">
        <Navbar />

        <div className="container mx-auto mt-20">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/services" element={<Services />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/Login" element={<Login />} />
            <Route path="/Signup" element={<Signup />} />
            <Route path="/Yieldprediction" element={<Yieldprediction />} />
            <Route path="/CropRecommendation" element={<CropRecommendation />} />
            <Route path="/Visualization" element={<Visualization />} />
            <Route path="/Profile" element={<Profile />} />
            <Route path="/DiseasePrediction" element={<DiseasePrediction />} />


          </Routes>
        </div>
        <Footer/>
        
      </div>
    </Router>
  );
};

export default App;
