COVID-19 Detection Using CT Scan Images
Overview
This project focuses on detecting COVID-19 from CT scan images using machine learning techniques. It applies advanced image preprocessing methods and builds an ensemble model combining multiple classifiers to improve prediction accuracy.
The system is designed to assist in early detection by analyzing medical imaging data efficiently and providing predictions through a user-friendly interface.

Objectives
* Perform preprocessing on CT scan images using histogram equalization techniques
* Reduce dimensionality using PCA and LDA
* Train multiple machine learning models
* Build an ensemble model for improved performance
* Deploy the model using a Streamlit web application

Methodology
Image Preprocessing
* Adaptive Histogram Equalization
* Histogram Equalization
* Conversion of images into structured datasets
Feature Engineering
* Principal Component Analysis (PCA)
* Linear Discriminant Analysis (LDA)
Models Used
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
Ensemble Learning
A soft voting classifier is used to combine predictions from:
* Random Forest
* KNN
* SVM
This improves overall model robustness and accuracy.

Tech Stack
* Python
* NumPy, Pandas
* Scikit-learn
* OpenCV
* Matplotlib / Seaborn
* Streamlit

Features
* End-to-end ML pipeline
* Image preprocessing + feature extraction
* Ensemble-based prediction system
* Interactive UI using Streamlit
* Modular and well-structured notebooks

Project Structure
project/
│── app.py
│── *.ipynb (model building & preprocessing)
│── .gitignore
│── README.md

How to Run

1. Clone the repository
git clone https://github.com/yourusername/covid-detection-ml.git
cd covid-detection-ml
2. Install dependencies
pip install -r requirements.txt
3. Run the application
streamlit run app.py

Results
* Improved prediction accuracy using ensemble learning
* Effective feature reduction using PCA & LDA
* Better classification performance compared to individual models

Note
* Dataset and image files are not included due to size limitations
* Model files are excluded to keep the repository lightweight

Future Enhancements
* Deploy on cloud (AWS / Heroku)
* Integrate deep learning (CNN models)
* Improve UI/UX of the application
* Add real-time image upload and prediction


Author
Joshna G

If you found this project useful
Give it a ⭐ on GitHub!
