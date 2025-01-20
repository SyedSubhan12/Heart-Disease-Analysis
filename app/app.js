// Import required modules
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { execSync } = require('child_process'); // Use child_process to run Python scripts for ML predictions

const app = express();
const PORT = 5000;

// Middleware
app.use(bodyParser.json());
app.use(cors());

// Path to the model file
const MODEL_PATH = 'D:/heart_disease_prediction/models/best_random_forest_model.pkl';

// API endpoint to handle predictions
app.post('/predict', async (req, res) => {
    try {
        const inputData = req.body.data;

        // Validate the input data
        if (!Array.isArray(inputData) || inputData.length === 0) {
            return res.status(400).json({ error: 'Invalid input data. Please provide a non-empty array.' });
        }

        // Reshape input data if needed (convert to comma-separated string)
        const reshapedInput = inputData.join(","); // Convert array to comma-separated string for Python input

        // Python command to make a prediction (adjust as per your model's needs)
        const predictCommand = `python -c "import joblib; model = joblib.load('${MODEL_PATH}'); input_data = [${reshapedInput}]; prediction = model.predict([input_data]); print(prediction[0])"`;

        // Run the Python script and get the prediction result
        const prediction = execSync(predictCommand).toString().trim();

        // Define categories based on your model's output
        const categories = [
            'No heart disease',
            'Mild heart disease',
            'Moderate heart disease',
            'Severe heart disease',
            'Very severe heart disease'
        ];

        // Map the prediction index to the corresponding category
        const predictionIndex = parseInt(prediction, 10);
        const predictionLabel = categories[predictionIndex] || 'Unknown';

        // Send the prediction response
        res.json({ prediction: predictionLabel, index: predictionIndex });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Error making prediction. Please try again later.' });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
