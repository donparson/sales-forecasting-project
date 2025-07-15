# Sales Forecasting for a Retail Chain

This project focuses on building a robust machine learning model to accurately forecast daily sales for a major retail chain across its various stores and product families. Leveraging historical sales data, promotional information, store characteristics, and holiday events, the aim is to provide actionable insights for inventory management, supply chain optimization, and marketing strategies.
## Business Problem

Accurate sales forecasting is critical for retail operations. Inaccurate forecasts can lead to stockouts, wasted inventory, inefficient resource allocation, and missed sales opportunities. This project addresses the challenge of predicting future sales with high precision to support strategic business decisions for a large Ecuadorian grocery retailer.
## Data Sources

The dataset used in this project is sourced from a Kaggle competition: [Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting). It includes:
* `train.csv`: Historical daily sales data for various stores and product families.
* `test.csv`: Future dates for which sales predictions are required.
* `stores.csv`: Metadata about each store (e.g., type, cluster, city).
* `holidays_events.csv`: Information on national, regional, and local holidays.
* `oil.csv`: Daily oil prices (though this dataset was not extensively used in the final model for this specific iteration, it was available).
## Methodology

The forecasting approach involved a comprehensive pipeline, from data preprocessing and extensive feature engineering to model training and evaluation using an XGBoost regressor.

### Data Preprocessing
* **Data Merging:** Combined `train.csv`/`test.csv` with `stores.csv` and `holidays_events.csv` on appropriate keys.
* **Date Handling:** Converted 'date' columns to datetime objects for time-series operations.
* **Missing Values:** Imputed missing values in `onpromotion` (with 0) and holiday-related columns (with 'No Holiday' or False for 'transferred').
* **Outlier Treatment:** Sales values less than 0 were capped at 0, as sales cannot be negative.

### Feature Engineering
A critical part of this project involved creating relevant features to capture temporal patterns and external influences:
* **Time-Based Features:** Extracted `year`, `month`, `day`, `dayofweek`, `dayofyear`, `weekofyear`, `quarter`, `is_month_start/end`, `is_year_start/end`, and `is_weekend`.
* **Lagged Sales:** Created features reflecting sales from previous periods (e.g., `sales_lag_7` for sales 7 days ago).
* **Rolling Averages:** Calculated rolling mean sales over different windows (e.g., `sales_rolling_mean_7`, `sales_rolling_mean_30`) to capture recent trends.
* **Promotional Information:** Utilized the `onpromotion` feature to capture the impact of promotional activities.
* **Categorical Encoding:** One-hot encoded categorical variables like `family`, `store_type`, `city`, `state`, `holiday_type`, `locale`, and `locale_name` to make them suitable for the XGBoost model.

### Model Selection & Training
* **Model:** **XGBoost Regressor** (`xgboost.XGBRegressor`) was chosen due to its robust performance on tabular data, ability to handle complex non-linear relationships, and built-in feature importance capabilities.
* **Time-Series Validation:** A **time-series split** was employed for validation, ensuring that the model was trained on past data and evaluated on future data (`train_df` up to 2017-07-15 for training, `val_df` from 2017-07-16 onwards for validation). This prevents data leakage and provides a more realistic assessment of future forecasting performance.
* ## Key Findings & Model Performance

The model demonstrated strong predictive capabilities on unseen validation data.

### Performance Metrics
* **Root Mean Squared Error (RMSE):** 526.23
* **Mean Absolute Error (MAE):** 145.74

These metrics indicate that the model provides reasonably accurate sales predictions, minimizing both the magnitude and squared magnitude of errors.

### Feature Importance
The feature importance analysis provided valuable insights into the primary drivers of sales:
* **`onpromotion`**: This feature consistently emerged as one of the most significant predictors, highlighting the direct impact of promotional activities on sales volume.
* **Time-Based Features (e.g., `dayofweek`, `weekofyear`, `month`)**: Seasonal patterns, weekly fluctuations, and yearly trends were highly influential, as expected in retail sales.
* **Lagged & Rolling Mean Sales (`sales_lag_7`, `sales_rolling_mean_7`)**: Past sales figures and recent sales trends proved to be extremely strong indicators of future sales, demonstrating the time-series dependency.
* **Store Characteristics (`store_type`, `cluster`, `city`)**: Differences in store demographics and locations played a notable role in sales variation.
  
<img width="918" height="526" alt="feature importance" src="https://github.com/user-attachments/assets/eb4deed6-3541-4783-bffa-d132b92ff029" />
 

### Residual Analysis
Analysis of residuals (the difference between actual and predicted sales) showed a relatively normal distribution centered around zero, suggesting the model captures most underlying patterns effectively. Some minor patterns related to extreme sales values were observed, indicating areas for potential refinement.

### Business Insights
* **Promotions are Key:** The model strongly affirms that promotional activities are a primary lever for driving sales.
* **Seasonality is Dominant:** Understanding weekly, monthly, and yearly sales cycles is crucial for inventory and staffing.
* **Historical Trends Matter:** Recent sales performance is a strong indicator, emphasizing the importance of timely data.
* **Store Specificity:** Sales are highly dependent on the type and location of the store, suggesting that targeted strategies per store cluster/type could be beneficial.
* ## Future Work & Potential Improvements

To further enhance the model's accuracy and robustness, the following areas could be explored:

1.  **Advanced Feature Engineering:**
    * Incorporate external economic indicators (e.g., inflation rates, unemployment) or weather data.
    * Develop more granular lagged features (e.g., lags specific to `family` *and* `store_nbr`).
    * Explore Fourier transform components to capture complex periodic patterns.
2.  **Hyperparameter Tuning:** Conduct more exhaustive hyperparameter optimization for XGBoost using techniques like GridSearchCV or RandomizedSearchCV.
3.  **Ensemble Modeling:** Combine XGBoost with other time-series models (e.g., Prophet, ARIMA) or other tree-based models to leverage their diverse strengths.
4.  **Outlier Treatment:** Implement more sophisticated methods for handling extreme sales outliers, as they can disproportionately affect model training.
5.  **Deep Learning Models:** Experiment with recurrent neural networks (RNNs) or Long Short-Term Memory (LSTM) networks for potentially better capture of long-term dependencies, especially if the dataset were larger or forecast horizon longer.
6.  **Probabilistic Forecasting:** Move beyond point forecasts to predict sales ranges or confidence intervals, providing a better understanding of uncertainty.
## Technologies Used

* **Python**
* **Pandas** (for data manipulation and analysis)
* **NumPy** (for numerical operations)
* **Scikit-learn** (for data splitting and metrics)
* **XGBoost** (for the machine learning model)
* **Matplotlib** (for plotting and visualization)
* **Seaborn** (for enhanced statistical data visualization)
* **JupyterLab** (for interactive development)
* **Git & Git LFS** (for version control and managing large files)
