import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { GiFarmer } from "react-icons/gi";
import { getAuth, signInWithEmailAndPassword, signOut } from "firebase/auth";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false); // Loading state
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  const auth = getAuth();

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((user) => {
      if (user) {
        setUser(user);
      } else {
        setUser(null);
      }
    });
    return unsubscribe;
  }, [auth]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null); // Reset previous errors

    try {
      await signInWithEmailAndPassword(auth, email, password);
      setTimeout(() => {
        setLoading(false);
        navigate("/"); // Redirect after login
      }, 3000); // Add slight delay for better UX
    } catch (err) {
      setLoading(false);
      setError("Invalid email or password. Please try again.");
    }
  };

  if (loading) {
    return (
      <div className="fixed inset-0 flex flex-col items-center justify-center bg-gray-900 z-50">
        <div className="animate-spin rounded-full h-16 w-16 border-4 border-white border-t-transparent"></div>
        <p className="text-white mt-6 text-md">Logging In...</p>
      </div>
    );
  }

  return (
    <div className="contact-container bg-gray-100 py-20 h-screen flex justify-center items-center">
      <div className="w-full max-w-md bg-white rounded-lg shadow-lg p-8">
        <div className="text-2xl flex justify-center items-center gap-2 font-bold mb-6">
          <GiFarmer className="text-primary text-5xl" />
          <p className="text-black text-4xl">AgriTech</p>
        </div>

        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-2 mb-4">
            <p>{error}</p>
          </div>
        )}

        {!user ? (
          <form onSubmit={handleLogin}>
            <div className="mb-4">
              <label htmlFor="email" className="block text-gray-700 text-lg font-semibold">
                Email
              </label>
              <input
                type="email"
                id="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:border-primary mt-2"
                required
              />
            </div>

            <div className="mb-6">
              <label htmlFor="password" className="block text-gray-700 text-lg font-semibold">
                Password
              </label>
              <input
                type="password"
                id="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-2 border-2 border-gray-300 rounded-md focus:outline-none focus:border-primary mt-2"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading} // Disable button when loading
              className={`w-full bg-primary text-white font-semibold rounded-md px-6 py-2 transition-all duration-200 ${
                loading ? "opacity-50 cursor-not-allowed" : "hover:bg-opacity-90"
              }`}
            >
              {loading ? "Logging In..." : "Login"}
            </button>
          </form>
        ) : (
          <div className="text-center">
            <p className="text-gray-700 text-lg mb-4">You are already logged in as {user.email}</p>
            <button
              onClick={() => signOut(auth).then(() => navigate("/"))}
              className="w-full bg-primary text-white font-semibold rounded-md px-6 py-2 transition-all duration-200 hover:bg-opacity-90"
            >
              Logout
            </button>
          </div>
        )}

        {!user && (
          <div className="mt-4 text-center text-gray-700">
            <p>
              Don't have an account?{" "}
              <Link to="/signup" className="text-primary font-semibold">
                Sign Up
              </Link>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Login;
