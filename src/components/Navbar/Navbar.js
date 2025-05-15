import React, { useState, useEffect } from "react";
import { GiFarmer } from "react-icons/gi";
import { MdMenu, MdClose } from "react-icons/md";
import { Link, useNavigate } from "react-router-dom"; 
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "../../firebase";

const Navbar = () => {
  const [open, setOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setIsLoggedIn(true);
        if (window.location.pathname === "/login" || window.location.pathname === "/signup") {
          navigate("/"); // Redirect only if the user was on login or signup
        }
      } else {
        setIsLoggedIn(false);
      }
    });

    return () => unsubscribe();
  }, [navigate]);

  const handleLogout = () => {
    setLoading(true);
    signOut(auth)
      .then(() => {
        setTimeout(() => {
          setIsLoggedIn(false);
          setLoading(false);
          navigate("/"); 
        }, 3000);
      })
      .catch((error) => {
        console.error("Error signing out: ", error);
        setLoading(false);
      });
  };

  if (loading) {
    return (
      <div className="fixed inset-0 flex flex-col items-center justify-center bg-gray-900 z-50">
        <div className="animate-spin rounded-full h-16 w-16 border-4 border-white border-t-transparent"></div>
        <p className="text-white mt-6 text-md">Logging Out....</p>
      </div>
    );
  }

  return (
    <nav className="fixed top-0 left-0 w-full bg-gray-900 shadow-md z-50 ">
      <div className="container flex justify-between items-center py-2 px-6 md:px-10">
        <div className="text-2xl flex items-center gap-2 font-bold">
          <GiFarmer className="text-primary text-5xl" />
          <p className="text-white">AgriTech</p>
        </div>

        <div className="hidden md:flex flex-1 justify-center">
          <ul className="flex items-center gap-6 md:gap-10 text-white text-lg">
            <li>
              <Link to="/" className="hover:text-primary duration-200">
                Home
              </Link>
            </li>
            <li>
              <Link to="/about" className="hover:text-primary duration-200">
                About
              </Link>
            </li>
            <li>
              <Link to="/services" className="hover:text-primary duration-200">
                Services
              </Link>
            </li>
            <li>
              <Link to="/contact" className="hover:text-primary duration-200">
                Contact
              </Link>
            </li>
          </ul>
        </div>

        <div className="hidden md:flex items-center gap-4">
          {isLoggedIn ? (
            <>
              <Link to="/profile">
                <button className="hover:bg-primary text-white font-semibold border-2 border-white px-6 py-2 rounded-lg transition-transform transform hover:scale-105">
                  Profile
                </button>
              </Link>
              <button
                onClick={handleLogout}
                className="hover:bg-primary text-white font-semibold border-2 border-white px-6 py-2 rounded-lg transition-transform transform hover:scale-105"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link to="/login">
                <button className="hover:bg-primary text-white font-semibold border-2 border-white px-6 py-2 rounded-lg transition-transform transform hover:scale-105">
                  Login
                </button>
              </Link>
              <Link to="/signup">
                <button className="hover:bg-primary text-white font-semibold border-2 border-white px-6 py-2 rounded-lg transition-transform transform hover:scale-105">
                  Signup
                </button>
              </Link>
            </>
          )}
        </div>

        <div className="md:hidden" onClick={() => setOpen(!open)}>
          {open ? (
            <MdClose className="text-4xl text-white" />
          ) : (
            <MdMenu className="text-4xl text-white" />
          )}
        </div>
      </div>

      {open && (
        <div className="md:hidden fixed top-16 left-0 w-full bg-white shadow-lg py-6 transition-all duration-300">
          <ul className="flex flex-col items-center gap-6 text-black text-lg">
            <li>
              <Link to="/" onClick={() => setOpen(false)}>
                Home
              </Link>
            </li>
            <li>
              <Link to="/about" onClick={() => setOpen(false)}>
                About
              </Link>
            </li>
            <li>
              <Link to="/services" onClick={() => setOpen(false)}>
                Services
              </Link>
            </li>
            <li>
              <Link to="/contact" onClick={() => setOpen(false)}>
                Contact
              </Link>
            </li>
            {isLoggedIn ? (
              <>
                <li>
                  <Link to="/profile" onClick={() => setOpen(false)}>
                    Profile
                  </Link>
                </li>
                <li>
                  <button
                    onClick={() => {
                      handleLogout();
                      setOpen(false);
                    }}
                  >
                    Logout
                  </button>
                </li>
              </>
            ) : (
              <>
                <li>
                  <Link to="/login" onClick={() => setOpen(false)}>
                    Login
                  </Link>
                </li>
                <li>
                  <Link to="/signup" onClick={() => setOpen(false)}>
                    Signup
                  </Link>
                </li>
              </>
            )}
          </ul>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
