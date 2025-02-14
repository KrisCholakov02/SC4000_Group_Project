# SC4000 - ELO Merchant Category Recommendation

## 1. Introduction

The **ELO Merchant Category Recommendation** challenge (originally from Kaggle) aims to predict a customer loyalty score based on transaction history and various merchant/card features. By accurately predicting loyalty scores, companies can design more targeted marketing campaigns, ultimately reducing cost and improving customer satisfaction.

### Key Objectives
1. **Predict loyalty scores** for each cardholder.
2. **Handle large, heterogeneous transaction data** (historical and new merchant transactions).
3. **Implement effective feature engineering** and outlier handling to improve model performance.
4. **Demonstrate a clean, scalable approach** suitable for production-level workflows.

---

## 2. What Makes Our Approach Novel

1. **Dual-Model Strategy with Outlier Detection**  
   - We trained a **binary classifier** (LightGBM) to distinguish outliers from typical data points.  
   - We then employed **two regression models**:  
     - One trained on *all* data (including outliers)  
     - Another trained only on the *non-outlier* subset  
   - During inference, if a test sample is classified as an outlier, we use the “all-data” regression model; otherwise, we use the “non-outlier” model.  
   - This targeted handling of outliers helped reduce the Root Mean Square Error (RMSE) and is relatively uncommon in standard Kaggle solutions.

2. **Extensive Feature Engineering**  
   - We created **basic statistical features** (count, sum, mean, std, etc.) from purchase amounts, dates, and installments.  
   - We introduced **behavioral features** (e.g., time gaps between purchases, recent activity, quantiles for purchase amounts).  
   - We utilized **second-order crossover features** to capture interactions among existing features.  
   - We used an **automated feature selection** approach (via a feature selection utility) to prune zero-importance features and reduce dimensionality without sacrificing performance.

3. **Efficient Hyperparameter Tuning**  
   - Hyperparameters such as number of leaves, feature fraction, and bagging fraction were systematically tuned using 5-fold cross-validation, ensuring that our final models generalize well.

4. **Competitive Results**  
   - Our final public Kaggle score was **3.68232** (top ~6.67% on the public leaderboard at the time of submission). This strong performance demonstrates that careful feature engineering and a dual-model strategy can rival more complex ensemble methods.

---

## 3. Repository Structure & File Descriptions

Below is an explanation of the key files in the repository and how they fit into the project pipeline.

1. **`.gitignore`**  
   - Standard Git configuration file specifying which files/folders should be ignored in version control (e.g., large data files, virtual environment folders, system files).

2. **`SC4000_Final_Report.pdf`**  
   - Your detailed final report explaining the full methodology, experimental results, EDA findings, and references.

3. **`eda.ipynb`**  
   - *Exploratory Data Analysis (EDA)* notebook.  
   - Provides initial data visualizations, statistical summaries, and correlation analyses.  
   - Useful for understanding data distributions, spotting anomalies, and guiding feature engineering.

4. **`data_cleaning.ipynb`**  
   - Notebook dedicated to **data cleaning** procedures, such as:  
     - Handling invalid or out-of-range values (e.g., `-1`, `999` for installments).  
     - Managing missing data.  
     - Splitting and transforming date features.  
   - Ensures a clean dataset ready for further processing.

5. **`data_processing.ipynb`**  
   - Focuses on **feature aggregation** and **transformation** tasks.  
   - Generates additional features (e.g., transaction counts, purchase date intervals, quantiles).  
   - Merges multiple sources (train, test, historical transactions, new transactions, merchant info) into a cohesive format for modeling.

6. **`elo-merchant-category-recommendation.ipynb`**  
   - The **main modeling notebook** (name may vary, but typically this is where the final pipeline is run).  
   - Trains and evaluates the LightGBM models for both regression (loyalty prediction) and classification (outlier detection).  
   - Produces predictions and exports results to CSV.

7. **`feature_selector.py`**  
   - Script or module that wraps around a **feature selection** utility (e.g., the FeatureSelector library).  
   - Identifies zero-importance features, calculates feature importance, and drops redundant features.

8. **`hyperparameter_tuning.ipynb`**  
   - Contains the **hyperparameter tuning** experiments, likely using a combination of grid search or Bayesian optimization with cross-validation.  
   - Finalizes the chosen LightGBM parameters (e.g., `num_leaves`, `feature_fraction`, `bagging_fraction`).

9. **`predictions_lgb.csv`**  
   - **Final prediction file** containing the model’s loyalty score predictions for the test set.  
   - Ready for Kaggle submission or offline evaluation.

---

## 4. Methodology Summary

1. **Data Cleaning**  
   - Invalid numeric values replaced or set to `NaN`.  
   - Missing categorical values imputed with placeholders.  
   - Date columns split into multiple granular features (day, week, month, year).

2. **Feature Engineering**  
   - Computed summary statistics (mean, std, min, max) on transaction-related features.  
   - Extracted behavioral features for both historical and new transactions.  
   - Built second-order interaction features (combinations of key features).  
   - Labeled extreme negative loyalty scores as “outliers” to handle them with a dedicated pipeline.

3. **Modeling**  
   - **Binary Classification**: Trained to detect whether a sample is an outlier.  
   - **Regression**: Two LightGBM regressors: one for all data (including outliers), another for non-outliers only.  
   - **Inference**: If a test point is flagged as an outlier (above a certain probability threshold), use the “all-data” regressor’s prediction; otherwise, use the “non-outlier” regressor.

4. **Hyperparameter Tuning**  
   - Systematic search for optimal `num_leaves`, `feature_fraction`, and `bagging_fraction` using 5-fold cross-validation.  
   - Early stopping to avoid overfitting and reduce computation time.

5. **Evaluation**  
   - RMSE on the validation set.  
   - Final Kaggle submission yielded a **3.68232** RMSE (top ~6.67% of participants).

---

## 5. Why This Project Stands Out

- **Comprehensive Data Pipeline**: Shows strong data cleaning, EDA, feature engineering, and model evaluation skills.  
- **Scalable Approach**: Uses LightGBM with large-scale data effectively; the pipeline is modular and can be adapted to real-world tasks.  
- **Novel Outlier Handling**: Demonstrates creativity and attention to detail by separately modeling outlier vs. non-outlier distributions.  
- **Strong Results**: Achieved a **top-tier** ranking on the public leaderboard, proving the approach’s effectiveness.

---

## 6. Usage & Future Improvements

- **Reproducibility**:  
  1. Clone this repository.  
  2. Place the relevant Kaggle data in the designated folders (ensure correct paths in notebooks).  
  3. Run `eda.ipynb` to explore the data.  
  4. Execute `data_cleaning.ipynb` and `data_processing.ipynb` for feature engineering.  
  5. Use `hyperparameter_tuning.ipynb` if you want to adjust model parameters.  
  6. Finally, run the main modeling notebook (e.g., `elo-merchant-category-recommendation.ipynb`) to train models and generate `predictions_lgb.csv`.

- **Potential Extensions**:  
  - **Neural Network Ensembles**: Incorporate MLPs or transformers for feature extraction.  
  - **Time-Series Analysis**: Explore sequence modeling (e.g., LSTM, TCN) for transactions over time.  
  - **Stacking or Blending**: Combine LightGBM with other algorithms (XGBoost, CatBoost) to potentially improve performance.

---

## 7. Conclusion

This project successfully tackles the ELO Merchant Category Recommendation challenge with a **focused, novel strategy**: outlier detection + two specialized regression models, backed by thorough feature engineering and hyperparameter tuning. The high leaderboard placement demonstrates the effectiveness of the approach. 

**Recruiters** and data science practitioners can see from this repository:
- A clean, modular codebase.
- Mastery of feature engineering and model selection.
- An intelligent handling of real-world data challenges (missing data, outliers, large dimensionality).

Feel free to explore the notebooks, tweak the hyperparameters, and use the methodology as a starting point for similar regression or ranking tasks in the future.

---

**Thank you for reviewing this project.** If you have any further questions or would like to discuss the approach in detail, please reach out!
