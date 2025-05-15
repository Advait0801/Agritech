import axios from "axios";

const API_URL = "http://127.0.0.1:5001";

export const predictDisease = async (image) => {
    const formData = new FormData();
    formData.append("file", image);

    return axios.post(`${API_URL}/predict_disease`, formData, {
        headers: {
            "Content-Type": "multipart/form-data",
        },
    });
};
