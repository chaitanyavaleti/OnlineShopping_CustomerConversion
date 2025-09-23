# OnlineShopping_CustomerConversion
This project builds a **Machine Learning (ML)** model to empower the business with data-driven insights to increase conversions, boost revenue, and enhance customer satisfaction and provides a **Streamlit app** for real-time predictions.

---

## 🚀 Features  
**Real-time predictions for conversion (classification).**
		Classification → Conversion
		Regression → Revenue
		Clustering → Customer segments
**Revenue estimation (regression).
**Display customer segments (clustering visualization).
**Show visualizations**
		Bar charts for categorical data or clusters
		Histograms for numerical predictions
		PCA scatter plot for clusters

## ⚙️ Setup Instructions  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/chaitanyavaleti/OnlineShopping_CustomerConversion.git
cd OnlineShopping_CustomerConversion
```
### 3️⃣ Run MLflow Tracking Server (Optional)  
```bash
mlflow ui
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 🏋️ Train Models with MLflow  

Run training script to:  
- Preprocess data  
- Train regression models  
- Log results with MLflow  
- Register best model automatically
			Classification
			Regression
			clustering

```bash
Online_Customer_Conversion_MLFlow.py
```

---

## 🌐 Run Streamlit App  

Run the app locally:  
```bash
streamlit run Customer_Conversion.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  


## 📊 Example Prediction  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **Scikit-learn, XGBoost** (ML models)  
- **MLflow** (experiment tracking & model registry)  
- **Streamlit** (frontend)  

## 👤 Created By

Chaitanya Valeti (MAE4 [AIML-C-WD-E-B18)
Built as a Mini project for the **AIML (Artificial Intelligence & Machine Learning)** domain at GUVI (HCL Tech).
