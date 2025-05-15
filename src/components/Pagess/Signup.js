import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { GiFarmer } from "react-icons/gi";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth, db } from "../../firebase"; // Ensure firebase is configured
import { collection, addDoc } from "firebase/firestore";
import { doc, setDoc } from "firebase/firestore"; // Import Firestore methods

const Signup = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData((prevState) => ({
      ...prevState,
      [e.target.name]: e.target.value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
  
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match!");
      return;
    }
  
    setLoading(true);
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );
      const user = userCredential.user;
  
      // Store user data in Firestore
      await setDoc(doc(db, "users", user.uid), {
        uid: user.uid,
        name: formData.name,
        email: formData.email,
      });
  
      navigate("/dashboard"); // Redirect after successful signup
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex justify-center items-center bg-gray-100 p-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow-lg p-8">
        <div className="text-2xl flex justify-center items-center gap-2 font-bold mb-6">
          <GiFarmer className="text-primary text-5xl" />
          <p className="text-black text-4xl">AgriTech</p>
        </div>

        {error && <p className="text-red-500 text-center mb-4">{error}</p>}

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-gray-700 text-lg font-semibold">
              Full Name
            </label>
            <input
              type="text"
              name="name"
              placeholder="Enter your full name"
              value={formData.name}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 text-lg font-semibold">
              Email
            </label>
            <input
              type="email"
              name="email"
              placeholder="Enter your email"
              value={formData.email}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 text-lg font-semibold">
              Password
            </label>
            <input
              type="password"
              name="password"
              placeholder="Create a password"
              value={formData.password}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              required
            />
          </div>

          <div className="mb-6">
            <label className="block text-gray-700 text-lg font-semibold">
              Confirm Password
            </label>
            <input
              type="password"
              name="confirmPassword"
              placeholder="Re-enter password"
              value={formData.confirmPassword}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full bg-primary text-white font-semibold rounded-md px-6 py-2 hover:bg-opacity-90"
            disabled={loading}
          >
            {loading ? "Signing up..." : "Sign Up"}
          </button>
        </form>

        <div className="mt-4 text-center text-gray-700">
          <p>
            Already have an account?{" "}
            <Link to="/login" className="text-primary font-semibold">
              Login
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Signup;
