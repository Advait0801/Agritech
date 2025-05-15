import axios from "axios";

const API_URL = "http://127.0.0.1:5001";

export const predict = async (data, token) => {
    return axios.post(`${API_URL}/predict_yield`, data, {
        headers: { Authorization: `Bearer ${token}` }
    });
    
};