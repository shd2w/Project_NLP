# Fake vs. Real News Detection in the Gaza-Israel Conflict

## Overview

The **Fake vs. Real News Detection in the Gaza-Israel Conflict** project employs advanced Natural Language Processing (NLP) techniques to distinguish between genuine news and propaganda within the context of the Gaza-Israel conflict. Utilizing a meticulously annotated dataset, the project leverages both traditional machine learning algorithms and state-of-the-art deep learning models to achieve high-accuracy classification. The primary objective is to develop an automated system that effectively combats misinformation and supports informed public discourse.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Data Preprocessing:** Comprehensive normalization and cleaning to prepare the dataset for analysis.
- **Class Imbalance Handling:** Utilizes advanced resampling techniques like SMOTE to address skewed class distributions.
- **Feature Engineering:** Employs TF-IDF vectorization, word embeddings, and contextual representations for robust feature extraction.
- **Model Development:** Implements a diverse set of machine learning and deep learning models, including ensemble methods and Large Language Model (LLM) integrations.
- **Performance Evaluation:** Assesses models using multiple metrics such as Accuracy, ROC AUC, Precision, Recall, and F1-Score.
- **Visualization:** Generates insightful visualizations, including word clouds and confusion matrices, to interpret model performance and data characteristics.

## Dataset

The dataset comprises **19,074** entries collected from various news platforms, including traditional media outlets, social media posts, and online news aggregators related to the Gaza-Israel conflict. Each entry is annotated with a label indicating the type of propaganda:

- **Propaganda:** Content deliberately designed to influence public perception.
- **Not Propaganda:** Genuine news without manipulative intent.
- **Unclear:** Ambiguous content where the presence of propaganda is uncertain.
- **Not Applicable:** Content irrelevant to propaganda classification.

**Data Source:** The dataset is sourced from reputable news organizations and social media platforms to ensure a comprehensive and unbiased representation of viewpoints.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Git**
- **Anaconda** (optional, for environment management)

### Clone the Repository

```bash
git clone [https://github.com/shd2w/Project_NLP.git]
cd fake-vs-real-news-gaza-israel
```

### Create a Virtual Environment (Optional but Recommended)

Using **Conda**:

```bash
conda create -n news-detection python=3.10
conda activate news-detection
```

Using **venv**:

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Ensure all dependencies are compatible with your system. If issues arise, consider using a Conda environment for better dependency management.

## Usage

The project is organized into two primary Jupyter Notebook files:

1. **Random Forest with Qwen Features_Random Forest with BERT Features_Logistic Regression with Qwen.ipynb**
2. **LR_SVM_NB_RF.ipynb**

### Running the Notebooks

1. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Navigate to the Project Directory:**

   Open the cloned repository folder in the Jupyter interface.

3. **Open and Execute the Notebooks:**

   - **Notebook 1:** `Random Forest with Qwen Features_Random Forest with BERT Features_Logistic Regression with Qwen.ipynb`
     - **Purpose:** Implements Random Forest classifiers enhanced with Qwen and BERT features, along with Logistic Regression models incorporating Qwen features.
   
   - **Notebook 2:** `LR_SVM_NB_RF.ipynb`
     - **Purpose:** Contains implementations of Logistic Regression (LR), Support Vector Machine (SVM), Naive Bayes (NB), and Random Forest (RF) models for propaganda classification.

4. **Review Results:**

   Each notebook includes sections on data preprocessing, feature engineering, model training, evaluation, and result visualization. Execute the cells sequentially to reproduce the analysis and obtain performance metrics.

## Project Structure

```
fake-vs-real-news-gaza-israel/
├── data/
│   └── Main.xlsx
├── notebooks/
│   ├── Random Forest with Qwen Features_Random Forest with BERT Features_Logistic Regression with Qwen.ipynb
│   └── LR_SVM_NB_RF.ipynb
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── ...
├── results/
│   ├── accuracy_metrics.png
│   └── confusion_matrix.png
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── utils.py
├── requirements.txt
├── README.md
└── LICENSE
```

- **data/**: Contains the dataset (`Main.xlsx`).
- **notebooks/**: Jupyter Notebooks for model development and analysis.
- **models/**: Serialized trained models for deployment or further analysis.
- **results/**: Visualizations and performance metric outputs.
- **src/**: Source code modules for data preprocessing, model training, and utility functions.
- **requirements.txt**: Lists all Python dependencies required for the project.
- **README.md**: Project documentation.
- **LICENSE**: Licensing information.

## Models

The project explores a variety of models to classify news content accurately:

### Traditional Machine Learning Models

- **Logistic Regression (LR):** A linear model suitable for binary and multiclass classification.
- **Support Vector Machine (SVM):** Effective for high-dimensional spaces and robust against overfitting.
- **Naive Bayes (NB):** A probabilistic classifier based on Bayes' theorem, effective for text classification.
- **Random Forest (RF):** An ensemble method leveraging multiple decision trees to improve classification accuracy.
- **XGBoost:** A gradient boosting framework known for its efficiency and performance.

### Deep Learning Models

- **Long Short-Term Memory (LSTM) with Attention Mechanism:** Captures long-term dependencies and focuses on relevant parts of the text.
- **Bidirectional LSTM:** Processes text in both forward and backward directions for comprehensive context understanding.
- **Transformer-Based Models (BERT, Qwen):** Leverages contextual embeddings for nuanced language understanding.

### Ensemble Methods

- **Stacked Classifier:** Combines multiple classifiers to enhance performance through diverse decision boundaries.
- **Voting Classifier:** Aggregates predictions from different models through majority voting to improve robustness.

## Results

### Model Performance

The following table summarizes the performance metrics achieved by each model:

| **Model**                                     | **Accuracy** | **ROC AUC** | **F1-Score (Macro)** |
|-----------------------------------------------|--------------|-------------|----------------------|
| Logistic Regression (LR)                      | 96.59%       | 0.9961      | 0.97                 |
| Naive Bayes (NB)                              | 96.14%       | 0.9910      | 0.96                 |
| Support Vector Machine (SVM)                  | 98.70%       | 0.9971      | 0.99                 |
| Random Forest (RF)                            | 98.94%       | 0.9996      | 0.99                 |
| XGBoost                                       | 97.85%       | 0.9954      | 0.98                 |
| Logistic Regression with BERT Features        | 85.30%       | 0.9205      | 0.85                 |
| Random Forest with BERT Features              | 87.40%       | 0.9302      | 0.86                 |
| Logistic Regression with Qwen Features        | 50.10%       | 0.7714      | 0.51                 |
| Random Forest with Qwen Features              | 56.64%       | 0.8154      | 0.58                 |
| LSTM with Attention Mechanism                 | 56.95%       | 0.4932      | 0.58                 |
| Bidirectional LSTM                            | 57.50%       | 0.4958      | 0.60                 |
| Stacked Classifier (SVM + RF)                 | 99.10%       | 0.9998      | 0.99                 |
| Voting Classifier (Majority Voting)           | 98.80%       | 0.9995      | 0.98                 |

### Key Findings

- **Random Forest (RF)** achieved the highest accuracy and ROC AUC score, closely followed by **Support Vector Machine (SVM)**.
- **Ensemble Methods** (Stacked Classifier) slightly outperformed individual traditional models, indicating the benefits of integrating multiple classifiers.
- **Deep Learning Models** and models incorporating **Large Language Model (LLM)** features (Qwen, BERT) underperformed compared to traditional machine learning models, highlighting potential challenges in feature integration and model tuning.
- **Error Analysis** revealed that models struggled with subtle manipulative language and ambiguous content, suggesting the need for more nuanced feature representations and model architectures.

## Contributing

Contributions are welcome! To contribute to this project, please follow the guidelines below:

1. **Fork the Repository:**

   Click the "Fork" button at the top right corner of this page.

2. **Clone Your Fork:**

   ```bash
   git clone https://github.com/yourusername/fake-vs-real-news-gaza-israel.git
   cd fake-vs-real-news-gaza-israel
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Changes:**

   Implement your feature or fix within the appropriate notebook or source file.

5. **Commit Changes:**

   ```bash
   git commit -m "Add Your Feature Description"
   ```

6. **Push to Your Fork:**

   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request:**

   Navigate to the original repository and create a pull request from your fork.

**Please ensure that your contributions adhere to the project's coding standards and include appropriate documentation and tests.**

## Acknowledgments

- **[NLTK](https://www.nltk.org/):** For providing robust NLP tools.
- **[Scikit-learn](https://scikit-learn.org/):** For the comprehensive machine learning library.
- **[TensorFlow](https://www.tensorflow.org/):** For the deep learning framework.
- **[Gensim](https://radimrehurek.com/gensim/):** For topic modeling and document similarity.
- **[Imbalanced-learn](https://imbalanced-learn.org/stable/):** For advanced resampling techniques.
- **[Hugging Face Transformers](https://huggingface.co/transformers/):** For state-of-the-art LLM implementations.
- **[OpenAI](https://openai.com/):** For developing powerful language models.
---
