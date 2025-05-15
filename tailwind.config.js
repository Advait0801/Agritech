/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#ff8901",
        secondary: "#fb923c",
        darkGray: "#2d2d2d",  // Custom dark gray for text
      },

      container: {
        center: true,
        padding: {
          DEFAULT: "1rem",
          sm: "2rem",
          lg: "4rem",
          xl: "5rem",
          "2xl": "6rem",
        },
      },

      fontFamily: {
        sans: ["Inter", "sans-serif"],  // Adding a modern font for a clean look
      },

      spacing: {
        18: "4.5rem",  // Custom spacing for responsive layout
      },

      screens: {
        sm: "640px",  // Small screens
        md: "768px",  // Medium screens
        lg: "1024px", // Large screens
        xl: "1280px", // Extra large screens
        "2xl": "1536px", // 2X Large screens
      },
    },
  },
  plugins: [],
};
