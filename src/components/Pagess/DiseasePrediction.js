import { useState } from "react";
import { predictDisease } from "../../API/DIseasePredictApi";

const DiseasePrediction = () => {
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleImageChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setImage(e.target.files[0]);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!image) {
            alert("Please upload an image");
            return;
        }

        setLoading(true);
        setError(null);
        setPrediction(null);

        try {
            const response = await predictDisease(image);
            setPrediction(response.data);
        } catch (err) {
            setError("Failed to get prediction. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex flex-col justify-center items-center bg-gray-100 py-12 px-4">
            <h2 className="text-4xl font-semibold text-center text-gray-800 mb-8">Disease Prediction</h2>
            <form onSubmit={handleSubmit} className="bg-white p-8 shadow-lg rounded-lg w-full max-w-lg space-y-6">
                <div>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageChange}
                        className="w-full p-3 border border-gray-300 rounded-md"
                    />
                </div>
                <button
                    type="submit"
                    className="w-full mt-6 bg-blue-600 text-white py-3 rounded-md hover:bg-blue-700 focus:outline-none"
                    disabled={loading}
                >
                    {loading ? "Processing..." : "Upload & Predict"}
                </button>
            </form>

            {error && <div className="text-red-500 mt-4">{error}</div>}

            {prediction && (
                <div className="mt-6 bg-white p-4 shadow-lg rounded-md w-full max-w-lg text-center">
                    <h3 className="text-xl font-semibold text-gray-800">Prediction Result</h3>
                    <p className="text-lg">Class: {prediction.predicted_class}</p>
                    <p className="text-lg">
                        Confidence Score: {(prediction.confidence_score * 100).toFixed(2)}%
                    </p>
                </div>
            )}
        </div>
    );
};

export default DiseasePrediction;
