# Anti-Pouching-CNN

# Forest Sound Classification Using CNN

## Overview
This project implements a deep learning model for classifying urban sound samples using the UrbanSound8K dataset. The application uses Mel-spectrogram feature extraction and a neural network to identify different urban sound classes.

## Features
- Audio file processing using librosa
- Mel-spectrogram feature extraction
- Deep learning neural network for sound classification
- Visualization of audio waveforms and spectrograms
- Prediction parser for individual audio files

## Prerequisites
- Python 3.7+
- Libraries:
  - librosa
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - keras
  - tensorflow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/urban-sound-classification.git
cd urban-sound-classification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
- Uses UrbanSound8K dataset
- Contains 10 urban sound categories
- Processed through fold-based classification

## Project Structure
```
urban-sound-classification/
│
├── data/
│   └── UrbanSound8K.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── notebooks/
│   └── audio_analysis.ipynb
└── README.md
```

## Key Components
- Mel-spectrogram feature extraction
- Sequential neural network with multiple dense layers
- Train-test split for model validation
- Custom prediction parser

## Model Architecture
- Input Layer: 128 features
- Hidden Layers: 
  - 1000 neurons (ReLU)
  - 750 neurons (ReLU)
  - 500 neurons (ReLU)
  - 250 neurons (ReLU)
  - 100 neurons (ReLU)
  - 50 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

## Usage

### Training the Model
```python
# Load data and preprocess
feature, label = load_and_preprocess_data()

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(feature, label)

# Train model
model.fit(X_train, Y_train, epochs=90, batch_size=50)
```

### Making Predictions
```python
# Predict sound for a single audio file
file_path = 'path/to/your/audiofile.wav'
prediction = prediction_parser(file_path, model)
```

## Visualization
- Waveform display
- Mel-frequency spectrograms
- Spectrogram with dynamic range in decibels

## Performance Metrics
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Your Name - [Your Email]

Project Link: [https://github.com/yourusername/urban-sound-classification](https://github.com/yourusername/urban-sound-classification)

## Acknowledgements
- UrbanSound8K Dataset
- Librosa Audio Processing Library
- Keras Deep Learning Framework
