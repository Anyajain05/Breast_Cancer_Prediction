# Breast Cancer Prediction

An end-to-end machine learning project for breast cancer diagnosis using the Wisconsin dataset.

## What is included

- Exploratory data analysis with matplotlib and seaborn
- Multiple classification algorithms with model comparison
- Best-model training and artifact saving
- Streamlit dashboard for EDA and interactive prediction

## Models Compared

- Logistic Regression
- Support Vector Machine
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting

## Local Setup

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. In Streamlit Cloud, connect the GitHub repository.
3. Set the app file path to `app.py`.
4. Deploy after the dependencies finish installing.

## Notes

- The dataset file is expected at `data.csv`.
- The app automatically trains and caches the best model if no saved artifact exists.