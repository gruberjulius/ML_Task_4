Project Page for the Machine Learning Class:

This paper and project explores the application of machine learning techniques to predict cumulative residual returns for approximately 500 stocks. The study focuses on data from 2010 to 2014 and examines several machine learning models, including Regularized Linear Regression, Random Forests, XGBoost, and LightGBM, to predict stock returns.

1. **Exploratory Data Analysis (EDA)**:
   - The target variable is the cumulative residual return for the next 24 hours, calculated daily at 15:30.
   - The target variable is approximately symmetric around zero, with significant outliers.
   - Data is standardized and clipped to reduce the influence of outliers.

2. **Feature Engineering**:
   - Features include technical indicators such as the Relative Strength Index (RSI) and Accumulation/Distribution (AD) indicator, along with other variables like rolling returns and volume changes.
   - The feature space is processed and normalized to enhance model performance.

3. **Model and Feature Selection**:
   - **Linear Models**: Lasso and Ridge regression were explored for their interpretability but were found to be less effective compared to tree-based models.
   - **Tree-Based Models**: Random Forest, XGBoost, and LightGBM were tested with various hyperparameters. XGBoost was identified as the most effective model, showing the best balance between avoiding overfitting and generalizing to new data.

4. **Cross-Validation and Model Performance**:
   - An expanding window cross-validation approach was used, with data from 2010-2013 for training and 2014 for validation.
   - The XGBoost model achieved the highest validation R² score, making it the final model of choice.
   - Feature importance was evaluated for each model, with indicators like Intraday RSI and cumulative residual returns consistently ranked high.

5. **Final Model and Results**:
   - The XGBoost model was selected as the final model due to its superior performance in validation.
   - A 30-day moving average of correlation was used to analyze feature relevance over time.
   - Predictions were found to be symmetric around zero, with the 2014 predictions showing a slightly negative bias.
   - Bin plots were used to visualize the distribution of returns across different years, revealing no clear seasonality patterns.

### Conclusion:
The study concludes that tree-based models, particularly XGBoost, are effective in predicting daily stock residual returns. The model showed promising results in validation but exhibited a slight negative bias in 2014. The research highlights the importance of feature selection and hyperparameter tuning in enhancing model performance.
