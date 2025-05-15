// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
//import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC8I_0iG56O9XzcUdV2i7-zDcimQ7ktu84",
  authDomain: window.location.hostname,
  projectId: "agritech-c7e39",
  storageBucket: "agritech-c7e39.firebasestorage.app",
  messagingSenderId: "561647456016",
  appId: "1:561647456016:web:6554162b352827a13c5f95",
  measurementId: "G-TPLDF72J1G"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
//const analytics = getAnalytics(app);


const auth = getAuth(app);
const db = getFirestore(app);


export { auth , db };