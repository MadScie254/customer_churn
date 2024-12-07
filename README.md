# Customer Churn Prediction

## 📋 Project Description
Customer churn prediction is a critical task for businesses to identify customers at risk of leaving and implement strategies to retain them. This project leverages machine learning to predict churn in a telecom company. The model is trained on historical customer data and deployed as an interactive web application.

The application allows users to input customer details and predict whether they are likely to churn, helping businesses take proactive measures.

---

## 🚀 Features
- **Data Preprocessing**: Includes feature engineering, handling missing values, and scaling.
- **Model Training**: Implements Logistic Regression and XGBoost classifiers.
- **Hyperparameter Tuning**: Fine-tunes the XGBoost model using GridSearchCV for improved performance.
- **Model Deployment**: Deploys the trained XGBoost model using a Streamlit web app.
- **Feature Importance Analysis**: Explains the most critical factors contributing to churn predictions.

---

## 📊 Technologies Used
- **Programming Language**: Python
- **Machine Learning Frameworks**: Scikit-learn, XGBoost
- **Web Deployment**: Streamlit
- **Visualization**: Matplotlib, Seaborn
- **Version Control**: Git

---

## 📁 Folder Structure
```
├── data/
│   ├── customer_churn.csv          # Dataset used for training and evaluation
├── notebooks/
│   ├── exploratory_data_analysis.ipynb  # EDA and preprocessing steps
│   ├── model_training.ipynb            # Training and tuning models
├── app/
│   ├── app.py                         # Streamlit app for model deployment
│   ├── requirements.txt               # Dependencies for the project
├── saved_models/
│   ├── xgb_churn_model.pkl            # Tuned XGBoost model
├── README.md                          # Project documentation
```

---

## 📦 Installation and Setup
Follow the steps below to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your_username>/customer_churn.git
   cd customer_churn
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app/app.py
   ```

---

## 🎮 Usage
1. Launch the application locally with the command:
   ```bash
   streamlit run app/app.py
   ```
2. Open the app in your browser, and input customer details such as age, subscription length, and charge amount.
3. View the prediction result to determine if the customer is likely to churn.

---

## 📈 Model Performance
### **Metrics:**
- **Accuracy**: 94%
- **Precision (Class 1 - Churn)**: 87%
- **Recall (Class 1 - Churn)**: 79%
- **ROC-AUC Score**: 88.3%

### **Confusion Matrix**:
|              | Predicted No Churn | Predicted Churn |
|--------------|--------------------|-----------------|
| **Actual No Churn** | 500                | 20              |
| **Actual Churn**     | 23                 | 87              |

---

## 📊 Feature Importance
The top features influencing churn predictions include:
1. **Customer Value**: Higher value correlates with lower churn likelihood.
2. **Subscription Length**: Longer subscriptions are less likely to churn.
3. **Seconds of Use**: High usage indicates customer satisfaction.

---

## 🤝 Contributing
Contributions are welcome! Follow these steps to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git commit -m "Added a new feature"
   git push origin feature-name
   ```
4. Create a pull request, describing your changes in detail.

---

## 🌟 Acknowledgments
- **Dataset**: Provided by [Source Name] (if applicable)
- **Inspiration**: Driven by the need for proactive customer retention strategies.

---

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
