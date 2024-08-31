
### **Predicting Loan Default using Machine Learning**
#### *Predictive Analysis of Loan Defaults Using ML**
**Project Overview**

This project aims to predict loan defaults using machine learning techniques. It involves preparing a dataset, feature selection, preprocessing, model training, evaluation, and improving model performance using different strategies, including SMOTE oversampling, feature scaling, normalization, and ensemble learning.

The primary aim of this project is to predict whether a loan will be fully paid back or will default. The outcome of this project is to develop a machine learning model that can accurately classify loans based on the likelihood of default.

**1\. Dataset and Problem Statement**

The dataset used for this project consists of various features related to loans, such as loan amount, interest rate, borrower details, and credit history. The primary objective is to classify loans as either "Fully Paid" (no default) or "Charged Off" (default) based on these features. The goal is to build a predictive model that can help lenders assess the risk associated with lending to different borrowers



**Reasons for Doing This Project:**

1. **Risk Assessment for Lenders**: By predicting loan defaults, financial institutions and lenders can better assess the risk associated with lending money to different borrowers. This helps in making informed decisions about loan approvals, interest rates, and loan amounts.
2. **Minimizing Financial Loss**: Accurately predicting which loans are likely to default allows lenders to take preventive measures, such as requiring higher collateral, adjusting interest rates, or declining risky loan applications. This reduces the chances of financial loss due to non-payment.
3. **Enhancing Credit Policies**: Understanding the factors that contribute to loan defaults can help institutions refine their credit policies and develop more effective strategies for managing credit risk.
4. **Improving Customer Selection**: By identifying characteristics and behaviors that are indicative of default, lenders can improve their customer selection process, targeting borrowers who are more likely to repay their loans.
5. **Supporting Regulatory Compliance**: Many financial institutions are required to adhere to regulatory standards for managing credit risk. A predictive model helps in meeting these compliance requirements by ensuring that lending practices are aligned with risk management guidelines.
6. **Automating Decision-Making**: Implementing a predictive model in the loan approval process can automate decision-making, making it faster and more efficient, which is beneficial for both the lender and the borrower.



**How It Works:**

1. **Input Data**: You provide the model with relevant information about the loan borrower. This can include details like:
    - **Loan amount**: The total amount of money requested by the borrower.
    - **Interest rate**: The percentage rate at which the loan is issued.
    - **Borrower’s financial details**: Information such as annual income, debt-to-income ratio, credit score, employment length, and home ownership status.
    - **Credit history**: Information about the borrower’s past loan repayments, number of credit lines, credit utilization, and any history of delinquencies or bankruptcies.
2. **Model Prediction**: The machine learning model uses this input data to analyze patterns based on what it learned during training. The model has been trained on historical loan data where it learned the features that are typically associated with loans that were fully paid back versus those that defaulted.
3. **Output**: The model provides a prediction of whether the borrower is likely to repay the loan ("Fully Paid") or default ("Charged Off"). This prediction helps lenders decide whether to approve the loan application and under what terms.



**Benefits of This Prediction:**

- **Informed Decision-Making**: Lenders can use the prediction to assess risk and make informed lending decisions.
- **Customizing Loan Offers**: Depending on the risk level, lenders can adjust loan terms (such as interest rates or repayment periods) to better match the borrower's risk profile.
- **Minimizing Losses**: By identifying high-risk borrowers, lenders can take precautions to reduce potential financial losses from loan defaults.



**Steps Involved in this project:**

**Data Cleaning**

- **Remove Unnecessary Loan Statuses**: Exclude loans that are currently active or have statuses other than "Fully Paid" or "Charged Off" to focus only on completed loans.
- **Map Target Variable**: Encode the target variable, loan_status, to binary values: "Fully Paid" as 0 and "Charged Off" as 1.

### **Feature Selection**

- **Identify Features with Multiple Unique Values**: Retain columns that have more than one unique value as they contribute valuable information for making predictions.
- **Drop Columns with Single Unique Value**: Remove columns that have only one unique value, as they do not provide predictive power.

**Handling Categorical Variables**

- **One-Hot Encoding**: Convert categorical variables into binary format using one-hot encoding to make them interpretable by machine learning models.

**Dealing with Missing Values**

- **Drop or Impute Missing Values**: Depending on the data, either drop rows with missing values or use statistical methods (mean, median, mode) to impute missing data. (This step was implied but not shown explicitly in the provided code.)

**Balancing the Dataset**

- **SMOTE Oversampling**: Apply Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance by oversampling the minority class (in this case, "Charged Off").

**Feature Scaling and Normalization**

- **Scaling and Normalization**: Scale and normalize numerical features to ensure all features contribute equally to the model, especially important for distance-based algorithms. (This step was mentioned but not shown explicitly in the provided code.)

**Splitting Data for Training and Testing**

- **Train-Test Split**: Split the data into training and testing sets to evaluate the model’s performance on unseen data.

**Model Building and Evaluation**

- Logistic Regression
- XGBoost Classifier
- Ensemble Learning and Voting Classifier

**Model Evaluation and Conclusion**

The final models are evaluated using metrics such as precision, recall, accuracy, and confusion matrix. The models are assessed for their ability to generalize to unseen data, and the results indicate the effectiveness of different preprocessing and modeling strategies in predicting loan defaults.


**Future Work**

Future work may include exploring additional feature engineering techniques, applying other oversampling and undersampling methods, tuning hyperparameters, and testing more complex models such as neural networks or deep learning algorithms. Further analysis of feature importance and SHAP (SHapley Additive exPlanations) values can also provide deeper insights into model decision-making processes.


**Conclusion**

This project demonstrates the application of machine learning techniques to financial data for predicting loan defaults. By carefully preprocessing data, selecting appropriate features, applying various modeling strategies, and rigorously evaluating model performance, this project provides a comprehensive approach to building predictive models for financial risk assessment.
