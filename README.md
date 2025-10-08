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
**With accuracy metrics**:

	{'Best Model': 'XGBoost', 'Score': 1.0, 'Registered As': 'Best_classification_Model', 'Version': 14}
	{'Best Model': 'Linear Regression', 'Score': 0.9999999930470421, 'Registered As': 'Best_regression_Model', 'Version': 25}
	{'Best Model': 'KMeans', 'Score': 0.2199326863104838, 'Registered As': 'Best_clustering_Model', 'Version': 42}
---

## 🌐 Run Streamlit App  

Run the app locally:  
```bash
streamlit run Customer_Conversion.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.  


## 📊 Example Prediction  

<img width="1853" height="855" alt="image" src="https://github.com/user-attachments/assets/499ec4ef-f0ff-4818-adf0-0a20e0878915" />

---

# Customer Segmentation Analysis

This project segments customers based on their online behavior patterns, including browsing, product interactions, and spending. Each cluster represents a distinct group of customers, enabling targeted marketing campaigns and personalized product recommendations.

---

## Cluster Interpretations

## Cluster Profiles

| Cluster | Price_2  | Page1_Main_Category | Model_Photography | Num_Customers |
|---------|----------|-------------------|-----------------|---------------|
| 0       | 1.593094 | 4                 | 1.725422        | 15522         |
| 1       | 2.000000 | 1                 | 1.000000        | 36780         |
| 2       | 1.529148 | 4                 | 2.000000        | 20310         |
| 3       | 1.000000 | 2                 | 1.000000        | 44534         |
| 4       | 1.520712 | 1                 | 1.187225        | 15233         |


### Cluster 0
- **Average Spending (`price_2`)**: 1.59 → Moderate spending  
- **Preferred Category (`page1_main_category`)**: 4  
- **Engagement (`model_photography`)**: 1.73 → Moderate engagement  
- **Number of Customers**: 15,522 → Medium-sized group  

### Cluster 1
- **Average Spending (`price_2`)**: 2.0 → Highest spending → **high-value buyers**  
- **Preferred Category (`page1_main_category`)**: 1  
- **Engagement (`model_photography`)**: 1.0 → Lower interaction, possibly focused buyers  
- **Number of Customers**: 36,780 → Large group  

### Cluster 2
- **Average Spending (`price_2`)**: 1.53 → Moderate spending  
- **Preferred Category (`page1_main_category`)**: 4 → Same as Cluster 0  
- **Engagement (`model_photography`)**: 2.0 → High interaction → engaged niche shoppers  
- **Number of Customers**: 20,310 → Medium-sized group  

### Cluster 3
- **Average Spending (`price_2`)**: 1.0 → Lowest spending → **browsers / low-value users**  
- **Preferred Category (`page1_main_category`)**: 2 → Different interest  
- **Engagement (`model_photography`)**: 1.0 → Low interaction  
- **Number of Customers**: 44,534 → Largest cluster  

### Cluster 4
- **Average Spending (`price_2`)**: 1.52 → Moderate spending  
- **Preferred Category (`page1_main_category`)**: 1 → Same as Cluster 1  
- **Engagement (`model_photography`)**: 1.19 → Slightly above minimal interaction  
- **Number of Customers**: 15,233 → Medium-sized group  

---

## Key Inferences

- **High-value buyers**: Cluster 1 (highest average spending) → prioritize premium offers and loyalty programs.  
- **Low-value / browsers**: Cluster 3 (lowest spending, largest group) → target with awareness campaigns and engagement initiatives.  
- **Engaged niche shoppers**: Cluster 2 → likely to interact with product pages/models; good candidates for targeted cross-selling.  
- **Category preferences**: Categories 1, 2, and 4 dominate across clusters → can guide category-specific marketing campaigns.  
- **Cluster sizes**: Large clusters represent general audience; smaller clusters indicate specialized or high-value segments for focused campaigns.  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **Scikit-learn, XGBoost** (ML models)  
- **MLflow** (experiment tracking & model registry)  
- **Streamlit** (frontend)  

## 👤 Created By

Chaitanya Valeti (MAE4 [AIML-C-WD-E-B18)
Built as a Mini project for the **AIML (Artificial Intelligence & Machine Learning)** domain at GUVI (HCL Tech).







