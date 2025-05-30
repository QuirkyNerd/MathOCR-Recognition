# MathOCR-Recognition

## Overview
HandwrittenMathRecognition is a Python-based project designed to recognize and solve handwritten mathematical expressions from images. The system takes an input image containing a math problem, extracts and recognizes individual characters, reconstructs the mathematical formula, computes the result, and visualizes the calculation process.

This project currently supports simple one-dimensional arithmetic expressions including addition, subtraction, multiplication, and division. The solution combines image preprocessing, character recognition using a convolutional neural network (LeNet-5), and formula parsing based on compiler theory techniques.

## Features
- Image preprocessing with OpenCV to isolate and normalize characters  
- Character recognition using a LeNet-5 CNN implemented in TensorFlow  
- Mathematical formula reconstruction using operator-precedence parsing and recursive descent parsing  
- Semantic understanding and evaluation of recognized expressions via attribute grammar value passing  
- Visualization of the calculation process and result using Matplotlib  
- Modular design enabling further extension to more complex formulas  

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/HandwrittenMathRecognition.git
cd HandwrittenMathRecognition
```
2. Create and activate a Python environment:
```bash
python -m venv env
source env/bin/activate # Windows: env\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```bash
HandwrittenMathRecognition/
│
├── input_images/ # Input images folder
├── models/ # Trained CNN models
├── src/ # Source code for preprocessing, recognition, parsing, evaluation
├── requirements.txt # Python dependencies
├── main.py # Entry point script
├── README.md # Project documentation
└── system_documentation.md # Detailed system instructions
```

## Technical Details
- Image preprocessing uses OpenCV for grayscale conversion, thresholding, contour detection, and character segmentation.  
- Character recognition is done with a LeNet-5 CNN trained on the CHROME dataset.  
- Formula parsing uses operator-precedence and recursive descent parsing methods from compiler theory.  
- Expression evaluation is implemented with attribute grammar value passing.  
- Visualization of calculation steps and results uses Matplotlib.
