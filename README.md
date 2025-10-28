# Customer Segmentation Using Machine Learning

A Streamlit web app that performs customer segmentation using K-Means clustering on customer data.

The goal is to group customers based on their behavior — such as age, gender, annual income, and spending score — to help businesses understand and target customer segments effectively.

---

## Overview

Customer segmentation is a powerful unsupervised learning technique used in marketing and business analytics.

This project uses K-Means clustering to find natural groupings in customer data and visualize them interactively.
The app allows users to:

* Upload a customer dataset (CSV)
* Automatically preprocess and clean the data
* Apply K-Means clustering
* Visualize results in beautiful 2D and 3D scatter plots
* Display cluster summary statistics

---

##  Features

*  Upload your own customer CSV file
*  Automated preprocessing and feature scaling
*  Dynamic K-Means clustering with adjustable cluster count
* 2D & 3D interactive visualizations
*  Cluster insights (means, sizes, etc.)
*  Built with Streamlit + Scikit-learn

---





##  Installation and Setup

#### 1️ Clone the Repository

```sh
git clone [https://github.com/chinnuhegde/Customer_Segmentation.git](https://github.com/chinnuhegde/Customer_Segmentation.git)
cd Customer_Segmentation

### 2️ Create a Virtual Environment

python -m venv venv
#Activate the environment:

#Windows:
venv\Scripts\activate
#macOS / Linux:
source venv/bin/activate

### 3️ Install Dependencies
pip install -r requirements.txt

### 4️ Run the Streamlit App
streamlit run Customer_Segmentation_app.py

### 5️ Open in Browser
Open your browser and go to: http://localhost:8501
