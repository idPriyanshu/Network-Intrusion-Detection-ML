
# 🚨 **Network Intrusion Detection Using Machine Learning** 🚨

## 📝 **Overview**

This repository demonstrates the use of **machine learning** techniques to analyze and detect cyber-attacks using a sample dataset (`dionaeaClean2.csv`). The dataset, collected from the **Dionea honeypot**, contains various features related to network traffic and potential attack activity. The project applies various **data preprocessing** steps, **exploratory data analysis (EDA)**, **dimensionality reduction**, **feature engineering**, and **machine learning models** to classify network traffic and predict potential attacks, such as **Denial-of-Service (DoS)** attacks.

The goal of this project is to showcase the application of **machine learning** in **cybersecurity** for attack detection, helping improve network security systems by automating attack prediction and classification.

---

## 🔧 **Skills & Knowledge Demonstrated**

- **Data Preprocessing**: Handling missing data, type conversions, and feature engineering for cleaner datasets.
- **Exploratory Data Analysis (EDA)**: Visualizing and analyzing the data to uncover patterns and correlations between different features using heatmaps and pair plots.
- **Dimensionality Reduction**: Reducing the feature space for better visualization and model performance with **PCA (Principal Component Analysis)**.
- **Machine Learning**: Building and training a **Random Forest Classifier** to detect cyber-attacks based on the provided dataset.
- **Model Evaluation**: Assessing model performance using **classification reports**, **confusion matrices**, and **accuracy metrics**.
- **Anomaly Detection**: Identifying and labeling suspicious source IPs based on request frequency and predicting attack types (DoS).
- **Time-Series Analysis**: Analyzing attack frequency over time to understand patterns and trends in attack behavior.

---

## 🧰 **Key Techniques and Libraries**

- **Pandas**: Data manipulation and cleaning (e.g., handling missing values, one-hot encoding, and timestamp conversions).
- **Matplotlib & Seaborn**: Data visualization tools used to generate **correlation matrices**, **scatter plots**, and **bar charts**.
- **Scikit-learn**: Machine learning library used for model training (**Random Forest Classifier**), dimensionality reduction (**PCA**), and model evaluation (**classification report**, **confusion matrix**).
- **Time-Series Analysis**: Analyzing attack frequency over time using `resample()` for time-based aggregation.

---

## 📂 **Project Structure**

```
├── dionaeaClean2.csv            # Sample dataset used for attack detection
├── attack_detection.ipynb       # Jupyter notebook containing the entire analysis pipeline
├── README.md                    # Project overview and description
└── requirements.txt             # Python dependencies
```

---

## 📊 **Dataset**

The dataset used in this project (`dionaeaClean2.csv`) contains network traffic data with the following key features:

- `src_ip`: Source IP address.
- `src_port`: Source port of the traffic.
- `dst_port`: Destination port of the traffic.
- `timestamp`: Timestamp of the network traffic.
- `type`: Type of network traffic (e.g., `accept`, `deny`).
- `request_count`: Number of requests made by the source IP.

This data was collected from a **Dionea honeypot**, which captures network traffic from attackers attempting to exploit vulnerabilities.

---

## 🛠️ **Project Steps**

1. **Data Preprocessing**: 
    - Cleaned up unnecessary columns and handled missing values.
    - Converted `src_port`, `dst_port`, and `timestamp` to the appropriate formats.
  
2. **Exploratory Data Analysis (EDA)**:
    - Visualized correlations between numeric features using a **heatmap**.
    - Used **pair plots** to analyze pairwise relationships among variables.

3. **Dimensionality Reduction**:
    - Applied **PCA** to reduce the data's dimensionality and visualize the results in 2D space.

4. **Machine Learning Model**:
    - Trained a **Random Forest Classifier** to predict `dst_port` and detect network anomalies.
    - Evaluated the model's performance using **classification reports** and **confusion matrices**.

5. **Attack Prediction**:
    - Classified source IPs as either involved in an attack or not based on request count.
    - Visualized the most targeted destination ports and identified patterns in the attack data.

6. **Time-Series Analysis**:
    - Resampled the data by date to visualize attack frequency over time.

---

## 🚀 **How to Run**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cybersecurity-attack-detection.git
   cd cybersecurity-attack-detection
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the `attack_detection.ipynb` notebook in Jupyter or **Google Colab**.

4. Run the notebook cell by cell to reproduce the results.

---

## 🧩 **Potential Applications**

This project is designed to be a starting point for building **network intrusion detection systems (IDS)** and could be extended in various ways:

- **Real-time Attack Detection**: Implement a real-time prediction system that flags suspicious traffic immediately.
- **Feature Expansion**: Incorporate more features such as **packet size**, **flow duration**, or **protocol type** to improve prediction accuracy.
- **Advanced Models**: Experiment with other machine learning models, such as **Support Vector Machines (SVM)**, **K-Nearest Neighbors (KNN)**, or **Deep Learning models**, for better classification performance.
- **Anomaly Detection**: Use **unsupervised learning techniques** like **Isolation Forests** or **DBSCAN** for anomaly detection in network traffic.

---

## 📋 **Requirements**

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can install the dependencies via:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🏁 **Conclusion**

This project demonstrates the potential of **machine learning** in **cybersecurity** to automate the detection of network anomalies and attacks. By leveraging the power of **classification algorithms** and **time-series analysis**, we can enhance network security systems and provide actionable insights for improving defense mechanisms.

---

## 📝 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Credits
**Developer:** [@idPriyanshu](https://www.github.com/idPriyanshu)  


---

## 📧 Contact
For queries or suggestions, email: [iiit.Priyanshu@gmail.com](mailto:iiit.priyanshu@gmail.com)
