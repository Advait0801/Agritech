import React, { useState, useEffect } from "react";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { doc, getDoc, setDoc } from "firebase/firestore";
import { db } from "../../firebase";
import { FaUserCircle } from "react-icons/fa";

const Profile = () => {
  const auth = getAuth();
  const [user, setUser] = useState(null);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    contact: "",
    gender: "",
    location: "",
    userType: "",
    photoURL: "",
  });
  const [initialData, setInitialData] = useState({}); // Stores original profile data
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);

  // Track authentication state
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe(); // Cleanup subscription
  }, []);

  // Fetch profile data from Firestore when user is set
  useEffect(() => {
    const fetchProfile = async () => {
      if (user) {
        try {
          const docRef = doc(db, "users", user.uid);
          const docSnap = await getDoc(docRef);
          if (docSnap.exists()) {
            setFormData(docSnap.data()); // Populate form with user data
            setInitialData(docSnap.data()); // Store initial data for reset
          } else {
            setFormData({ ...formData, email: user.email }); // Set email if no profile exists
          }
        } catch (error) {
          console.error("Error fetching profile:", error);
        }
      }
    };
    fetchProfile();
  }, [user]);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle file changes for the profile picture
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setFormData({ ...formData, photoURL: reader.result });
      };
      reader.readAsDataURL(file);
    }
  };

  // Save the updated profile to Firestore
  const handleSubmit = async (e) => {
    e.preventDefault();
    e.stopPropagation(); // Prevent unintended form submission

    console.log("Saving Profile...");
    setLoading(true);
    try {
      await setDoc(doc(db, "users", user.uid), formData, { merge: true });
      setEditing(false);
      setInitialData(formData); // Update initial data after save
    } catch (error) {
      console.error("Error updating profile:", error);
    }
    setLoading(false);
  };

  // Reset form data to the initial fetched values
  const handleCancel = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setEditing(false);
    setFormData(initialData);
  };

  return (
    <div className="min-h-screen flex justify-center items-center bg-gray-100 p-4">
      <div className="w-full max-w-lg bg-white rounded-lg shadow-lg p-6">
        <div className="text-center">
          <div className="relative inline-block w-32 h-32 rounded-full overflow-hidden border-4 border-primary">
            {formData.photoURL ? (
              <img
                src={formData.photoURL}
                alt="Profile"
                className="w-full h-full object-cover"
              />
            ) : (
              <FaUserCircle className="w-full h-full text-gray-400" />
            )}
            {editing && (
              <input
                type="file"
                accept="image/*"
                className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
                onChange={handleFileChange}
              />
            )}
          </div>
          <h2 className="text-2xl font-semibold text-black mt-4">
            {formData.name || "Your Name"}
          </h2>
          <p className="text-gray-600">{formData.email}</p>
        </div>

        <form onSubmit={handleSubmit} className="mt-6">
          <div className="mb-4">
            <label className="block text-gray-700 font-semibold">Full Name</label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              disabled={!editing}
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 font-semibold">Contact No</label>
            <input
              type="text"
              name="contact"
              value={formData.contact}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              disabled={!editing}
              required
            />
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 font-semibold">Gender</label>
            <select
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              disabled={!editing}
              required
            >
              <option value="">Select Gender</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 font-semibold">User Type</label>
            <select
              name="userType"
              value={formData.userType}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              disabled={!editing}
              required
            >
              <option value="">Select User Type</option>
              <option value="Student">Student</option>
              <option value="Farmer">Farmer</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="mb-4">
            <label className="block text-gray-700 font-semibold">Location</label>
            <input
              type="text"
              name="location"
              value={formData.location}
              onChange={handleChange}
              className="w-full px-4 py-2 border rounded-md focus:border-primary"
              disabled={!editing}
              required
            />
          </div>

          <div className="flex justify-between gap-2">
            {!editing ? (
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log("Edit Mode Activated");
                  setEditing(true);
                }}
                className="w-full bg-gray-700 text-white font-semibold rounded-md px-6 py-2 hover:bg-gray-600"
              >
                Edit Profile
              </button>
            ) : (
              <>
                <button
                  type="submit"
                  className="w-1/2 bg-primary text-white font-semibold rounded-md px-6 py-2 hover:bg-opacity-90"
                >
                  {loading ? "Saving..." : "Save Profile"}
                </button>
                <button
                  type="button"
                  onClick={handleCancel}
                  className="w-1/2 bg-gray-700 text-white font-semibold rounded-md px-6 py-2 hover:bg-gray-600"
                >
                  Cancel
                </button>
              </>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default Profile;
