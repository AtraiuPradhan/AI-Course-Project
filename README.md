# AI-Course-Project
Description of AI Course Project for Reg No - 23BCE0872
# Web-Based AI Image Classifier

A lightweight, web-based image classification system that identifies common objects such as **dogs, cats, cars, bicycles, bottles, and fruits** using a **pre-trained MobileNetV2 deep learning model**.  
The project demonstrates real-time AI inference through a simple and interactive **Streamlit** interface.


## Features
- Upload any `.jpg`, `.jpeg`, or `.png` image
- Real-time object classification using MobileNetV2 (ImageNet pre-trained)
- Displays **Top-3 predictions** with confidence scores
- Shows **bar chart visualization** for prediction confidence
- Lightweight, runs locally (no GPU or cloud dependency)
- Fully open-source and beginner-friendly


## Project Structure
AI-Course-Project/
│
├── app.py # Main Streamlit web app
├── requirements.txt # List of dependencies
│
├── report/
│ ├── 23BCE0872_AI Course Project.pdf # Final report (ACL-style)
│ ├── 23BCE0872_AI Course Project.tex # LaTeX source
│
├── sample_images/ #Optional demo images
│
└── README.md # This file


## Installation & Setup
### Clone the repository
git clone https://github.com/AtraiuPradhan/AI-Course-Project.git
cd AI-Course-Project
### Install dependencies
pip install -r requirements.txt
### Run the Streamlit app
streamlit run app.py
