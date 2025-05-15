import React, { useState, useEffect } from "react";
import { auth, db } from "../../firebase";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";

const Contact = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });
  const [user, setUser] = useState(null);
  

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!user) {
      alert("You need to be logged in to submit the form.");
      return;
    }

    try {
      await addDoc(collection(db, "messages"), {
        name: formData.name,
        email: formData.email,
        message: formData.message,
        userId: user.uid,
        timestamp: serverTimestamp(), // Server-generated timestamp
      });
      alert("Message sent successfully!");
      setFormData({ name: "", email: "", message: "" });
    } catch (error) {
      console.error("Error adding document: ", error);
      alert("Error sending message. Try again later.");
    }
  };

  return (
    <div className="contact-container bg-gray-100 py-20">
      <section className="contact-header text-center px-4">
        <p className="text-4xl font-bold text-black">Get in Touch with us at</p>
        
        <br />
        <p className="text-6xl font-bold text-primary">AgriTech</p>
        <br />
        <p className="mt-4 text-lg text-gray-700">
          Have questions or want to learn more about our Agri Tech solutions?
          Reach out to us!
        </p>
      </section>

      <section className="contact-form mt-20 max-w-4xl mx-auto px-4">
        <h2 className="text-3xl font-semibold text-center">Send Us a Message</h2>
        <form
          onSubmit={handleSubmit}
          className="bg-white p-8 mt-8 rounded-lg shadow-lg"
        >
          <div className="mb-4">
            <label htmlFor="name" className="block text-sm font-semibold">
              Your Name
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              className="w-full p-3 mt-1 border border-gray-300 rounded-md"
              placeholder="Enter your name"
              required
            />
          </div>
          <div className="mb-4">
            <label htmlFor="email" className="block text-sm font-semibold">
              Your Email
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="w-full p-3 mt-1 border border-gray-300 rounded-md"
              placeholder="Enter your email"
              required
            />
          </div>
          <div className="mb-4">
            <label htmlFor="message" className="block text-sm font-semibold">
              Your Message
            </label>
            <textarea
              id="message"
              name="message"
              value={formData.message}
              onChange={handleChange}
              className="w-full p-3 mt-1 border border-gray-300 rounded-md"
              rows="5"
              placeholder="Your message here"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-primary text-white py-3 rounded-md mt-4 hover:bg-secondary"
            
          >
            Submit
          </button>
        </form>
      </section>
    </div>
  );
};

export default Contact;
