# Credit Risk Probability Model for Alternative Data

## Project Overview
This project implements an end-to-end credit scoring system using alternative data from e-commerce transactions. The model transforms behavioral patterns into predictive risk signals for "Buy Now, Pay Later" services.

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

**Basel II Influence:**
The Basel II Capital Accord emphasizes three pillars: Minimum Capital Requirements, Supervisory Review, and Market Discipline. For our model:

- **Pillar 1 (Minimum Capital Requirements):** Requires banks to hold capital proportional to their credit risk exposure. Our model directly informs this by quantifying risk probabilities.
- **Pillar 2 (Supervisory Review):** Regulators must ensure banks have robust internal processes. This necessitates:
  - **Explainability:** Clear documentation of how risk scores are derived
  - **Backtesting:** Regular validation of model predictions
  - **Stress testing:** Assessment under adverse conditions
- **Pillar 3 (Market Discipline):** Requires transparent disclosure, making interpretable models essential for stakeholder trust.

**Impact on Our Model:**
We need an interpretable model because:
- Regulatory compliance requires understanding of risk drivers
- Loan officers need to explain decisions to customers
- Model validation and auditing demand transparency
- Capital allocation decisions require justifiable risk assessments

### 2. Proxy Variable Necessity and Risks

**Why a Proxy is Necessary:**
1. **Data Limitation:** E-commerce platforms don't track loan repayments
2. **Behavioral Correlation:** Purchase patterns correlate with financial responsibility
3. **Innovation Opportunity:** Alternative data enables credit access for underserved populations

**Business Risks of Proxy-Based Predictions:**

| Risk Category | Description | Mitigation Strategy |
|--------------|-------------|-------------------|
| **Misalignment Risk** | Proxy may not perfectly correlate with actual default | - Validate with small pilot programs<br>- Monitor performance metrics closely<br>- Use conservative initial thresholds |
| **Regulatory Risk** | Regulators may question proxy validity | - Document rationale thoroughly<br>- Cite academic research on RFM-credit correlation<br>- Maintain audit trails |
| **Discrimination Risk** | Proxy may inadvertently discriminate | - Test for bias across protected classes<br>- Implement fairness constraints<br>- Regular bias audits |
| **Model Decay Risk** | Relationship between proxy and default may change | - Continuous monitoring<br>- Regular retraining<br>- Set up drift detection |

**Specific Proxy Approach:**
We'll use **RFM-based clustering** to identify "disengaged" customers as high-risk proxies, based on the hypothesis that consistent, high-value engagement indicates financial stability.

### 3. Model Selection Trade-offs

| Aspect | Simple Model (Logistic + WoE) | Complex Model (Gradient Boosting) |
|--------|-------------------------------|-----------------------------------|
| **Interpretability** |  **High** - Clear coefficients, WoE provides business intuition | ❌ **Low** - Black-box nature, feature importance but not causality |
| **Regulatory Compliance** |  **Easier** - Transparent decision process | ❓ **Challenging** - May require extensive validation and explanation |
| **Performance** |  **Moderate** - May miss complex interactions |  **High** - Captures non-linear relationships well |
| **Implementation Cost** |  **Low** - Easier to build, test, and maintain |  **High** - Requires more computational resources |
| **Risk Management** |  **Better** - Clear risk drivers facilitate stress testing |  **Complex** - Harder to identify failure modes |
| **Adaptability** |  **Limited** - Assumes linear relationships |  **Better** - Adapts to changing patterns |

**Recommended Hybrid Approach:**
1. **Start with interpretable models** for regulatory approval and stakeholder trust
2. **Use complex models** for initial screening where interpretability is less critical
3. **Implement model ensembles** that balance both needs
4. **Maintain parallel models** for comparison and validation

## Business Justification for Our Approach

### Why RFM as a Credit Risk Proxy:
1. **Recency:** Recent activity suggests ongoing financial engagement
2. **Frequency:** Regular transactions indicate stable income patterns
3. **Monetary:** Spending levels correlate with financial capacity

### Regulatory Alignment:
- **Transparent methodology:** RFM is well-understood in marketing, adaptable to credit
- **Audit-friendly:** Each component is easily traceable to raw data
- **Risk-based:** Aligns with Basel's emphasis on risk-sensitive approaches

### Implementation Strategy:
1. **Phase 1:** Deploy conservative model with high interpretability
2. **Phase 2:** Gather performance data on actual defaults
3. **Phase 3:** Refine model using real default labels when available
4. **Phase 4:** Gradually introduce complexity while maintaining oversight

---

## Project Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (for deployment)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
## Task 2: Exploratory Data Analysis 

### Objective
The objective of this task is to explore and understand the eCommerce transaction dataset in order to uncover behavioral patterns, data quality issues, and statistical characteristics that will inform feature engineering and proxy credit risk definition in later stages of the project.

Since no explicit loan default label exists, this exploratory analysis focuses on understanding customer transaction behavior, variability, and risk-related signals that can be used to construct a proxy for credit risk.

---

### Methodology

The EDA was conducted using a structured and systematic approach:

1. **Data Overview & Quality Checks**
   - Verified dataset size, structure, and data types
   - Checked for missing values and duplicate records
   - Assessed overall data cleanliness and reliability

2. **Descriptive Statistics**
   - Computed summary statistics (mean, median, standard deviation, skewness, kurtosis)
   - Analyzed distribution shape and variability of numerical features

3. **Distribution Analysis**
   - Visualized numerical feature distributions to identify skewness and heavy tails
   - Examined categorical feature frequencies to understand behavioral preferences

4. **Outlier Detection**
   - Applied multiple outlier detection techniques (IQR, Z-score, percentile)
   - Quantified the impact of extreme values on transaction behavior and fraud occurrence

5. **Customer Behavior Analysis**
   - Analyzed transaction frequency and volume at the customer level
   - Identified highly active vs low-activity customers

6. **Temporal Pattern Analysis**
   - Investigated transaction trends across hours, days, and months

7. **Correlation Analysis**
   - Evaluated relationships between numerical features
   - Identified highly correlated and redundant variables

---

### Key Findings

#### 1. **Exceptional Data Quality**
- **95,662 transactions** across **16 features**
- **0% missing values** and **0% duplicate rows**
- **3,742 unique customers** with average 26 transactions each
- **90-day time period** (Nov 2018 - Feb 2019)

#### 2. **Critical Risk Indicators Discovered**
- **Extreme Transaction Variability**: Coefficient of Variation = 1,835.5%
- **Right-Skewed Distribution**: Skewness = 51.10 (Median: $1,000 vs Mean: $6,717)
- **Heavy-Tailed Distribution**: Kurtosis = 3,363 (many extreme outliers)
- **Fraud Concentration**: Outlier transactions have 4x higher fraud risk

#### 3. **Customer Behavior Patterns**
- **Pareto Distribution**: Top 10% of customers make 25%+ of transactions
- **Segmentation Clear**: 50% of transactions ≤ $1,000, 1% > $80,000
- **Product Preference**: Category '3' dominates (68.9% of transactions)

#### 4. **Statistical Insights for Modeling**
- **Amount-Value Correlation**: 0.990 (near-perfect, indicates credits exist)
- **Fraud Class Imbalance**: Overall rate = 0.20%, but 20% in large transactions
- **Temporal Patterns**: Clear daily/hourly transaction patterns

### Key Insight

1. **Extreme Transaction Variability**
   - Transaction amounts exhibit very high variability (CV > 1800%) and extreme right skewness.
   - The mean transaction value is significantly higher than the median, indicating strong outlier influence.
   - This suggests the need for log transformation and robust aggregation techniques during feature engineering.

2. **Distinct Customer Behavior Segments**
   - Customer activity is highly uneven: a small proportion of customers accounts for a large share of transactions.
   - Median customers have low transaction frequency, while a few customers transact thousands of times.
   - This strongly supports the use of RFM (Recency, Frequency, Monetary) analysis for risk segmentation.

3. **Fraud Is Rare but Concentrated**
   - Fraudulent transactions are extremely rare overall but are disproportionately concentrated among large-value transactions.
   - Outlier transactions are several times more likely to be associated with fraud.
   - This highlights the importance of handling class imbalance and extreme values carefully.

4. **Redundant Monetary Features**
   - `Amount` and `Value` are almost perfectly correlated, indicating that `Value` represents the absolute transaction amount.
   - Negative transaction amounts reveal refunds or reversals, which must be treated explicitly in feature engineering.

5. **Behavioral Signals in Categorical and Temporal Features**
   - Product category and channel usage show dominant patterns that may reflect customer risk profiles.
   - Temporal features alone have weak correlations but may add value when combined with behavioral aggregates.

---

### Implications for Credit Risk Modeling

####  Positive Indicators:
- Clean, complete dataset with sufficient volume
- Clear patterns for customer segmentation
- Multiple features for behavioral analysis

- Customer transaction behavior provides meaningful signals for constructing a proxy credit risk label.
- High transaction variability, low engagement, and inconsistent spending patterns may indicate higher credit risk.
- The findings directly inform:
  - Feature aggregation strategies (Task 3)
  - RFM-based customer clustering for proxy target creation (Task 4)
  - Model selection and preprocessing decisions (Task 5)

This exploratory analysis establishes a strong foundation for building an interpretable, behavior-driven credit risk model in the absence of explicit default labels.
#### ⚠️ Challenges Identified:
- Extreme outliers require robust handling
- Severe class imbalance in fraud detection
- Right-skewed distributions need transformation

#### 🎯 Feature Engineering Direction:
1. **Create RFM metrics** per customer (Recency, Frequency, Monetary)
2. **Transform Amount/Value** using log transformation
3. **Engineer variability features** (CV, IQR, percentile ratios)
4. **Extract temporal patterns** from transaction times
5. **Create categorical embeddings** for product/channel preferences


## Task 3: Feature Engineering - Implementation Details

### 📋 Objective
Build a robust, automated, and reproducible data processing script that transforms raw data into a model-ready format using sklearn.pipeline.Pipeline.

### 🏗️ Architecture
Implemented all 6 required feature engineering components as sklearn-compatible OOP classes:

### 🔧 Key Components Implemented

#### 1. **Aggregate Features** (`AggregateFeatures` class)
- Calculated customer-level statistics from transaction data
- **Features created:**
  - `TotalAmount`: Sum of all transaction amounts per customer
  - `AvgAmount`: Average transaction amount per customer  
  - `TransactionCount`: Number of transactions per customer
  - `StdAmount`: Standard deviation of transaction amounts
  - `MinAmount`, `MaxAmount`, `MedianAmount`: Additional statistics

#### 2. **Temporal Features** (`TemporalFeatureExtractor` class)
- Extracted time-based patterns from TransactionStartTime
- **Features created:**
  - `TransactionHour`, `TransactionDay`, `TransactionMonth`, `TransactionYear`
  - `TransactionDayOfWeek`, `TransactionWeekOfYear`, `IsWeekend`

#### 3. **Categorical Encoding** (`CategoricalEncoder` class)
- Converted categorical variables to numerical format
- **Strategies implemented:** One-Hot Encoding and Label Encoding
- **Encoded columns:** ProductCategory (9 categories) and ChannelId (4 channels)

#### 4. **Missing Value Handling** (`MissingValueHandler` class)
- Identified and imputed missing values
- **Strategy used:** Median imputation for `StdAmount` (712 missing values, 0.74%)
- **Result:** Zero missing values after processing

#### 5. **Feature Scaling** (`FeatureScaler` class)
- Normalized numerical features to common scale
- **Strategy used:** Standardization (mean=0, std=1)
- **Scaled features:** 17 numerical features including Amount, Value, and RFM metrics

#### 6. **WoE and IV Transformation** (`WOETransformer` class)
- Calculated Weight of Evidence (WoE) and Information Value (IV)
- **Demonstrated using FraudResult** (Note: Will be replaced with RFM-based proxy in Task 4)
- **IV Analysis revealed:**
  - Amount, Value, and RFM metrics show suspiciously high IV (>0.5)
  - Temporal features show weak to medium predictive power

### 📊 Results & Output

#### Data Transformation Pipeline:

#### Key Statistics:
- **Original features:** 16 columns
- **Engineered features:** 40 columns (150% increase)
- **Feature types:**
  - Temporal features: 8
  - Encoded categorical: 13  
  - Original features: 7
  - Aggregate RFM: 7
  - WoE features: 5

#### Data Quality:
- **Missing values:** 0% (imputed successfully)
- **Duplicates:** 0% (maintained data integrity)
- **Memory optimized:** Parquet file (2.12 MB) vs CSV (73.04 MB)

### 🔍 Insights from Feature Engineering

1. **Customer Segmentation Revealed**: RFM metrics show clear customer tiers
   - Top customer (CustomerId_4406): 119 transactions, $109,921 total
   - Average transaction size varies significantly ($923.71 mean vs $1,000 median)

2. **Temporal Patterns Identified**: Transaction peaks at specific hours/days
   - Hourly distribution shows business hour concentrations
   - Weekend vs weekday patterns evident

3. **Data Scaling Required**: Extreme value ranges necessitated standardization
   - Amount: Range -$1M to $9.88M → Standardized to mean=0, std=1
   - Value: Similar transformation applied

4. **WoE Analysis Insightful**: 
   - High IV values suggest strong predictive power in transaction amounts
   - Temporal features provide moderate predictive signals

   ## Task 4: Proxy Target Variable Engineering - Implementation Details

### 📋 Objective
Create a credit risk proxy target variable since we don't have a pre-existing "credit risk" column in the data. Programmatically identify "disengaged" customers and label them as high-risk proxies using RFM analysis and clustering.

### 🎯 The 4 Required Steps Implemented:

#### **Step 1: Calculate RFM Metrics** (`RFMMetricsCalculator` class)
- **Recency**: Days since last transaction (from consistent snapshot date)
- **Frequency**: Number of transactions per customer
- **Monetary**: Total positive amount spent per customer
- **Additional**: Log-transformed versions for clustering

#### **Step 2: Cluster Customers** (`RFMClustering` class)
- Used **K-Means algorithm** to segment customers into **3 distinct groups**
- **Proper pre-processing**: Log transformation + StandardScaler for meaningful results
- **Reproducibility**: Set `random_state=42` as per instruction
- **Quality metric**: Silhouette Score = 0.3328

#### **Step 3: Define High-Risk Label** (`HighRiskLabelAssigner` class)
- Analyzed clusters to identify **least engaged segment** (high recency, low frequency, low monetary)
- Created binary target column `is_high_risk`
- **Risk identification**: Cluster 0 identified as high-risk (1117 customers, 29.85%)

#### **Step 4: Integrate Target Variable** (`TargetVariableIntegrator` class)
- Merged `is_high_risk` column back to main processed dataset
- Handled CustomerId format mismatches between raw and engineered data
- Created customer-risk mapping for reference

### 📊 Results & Analysis

#### **RFM Statistics:**
- Total customers: 3,742
- Recency: Mean = 31.5 days, Range = 1-91 days
- Frequency: Mean = 25.6 transactions, Range = 1-4,091 transactions
- Monetary: Mean = $211,864, Range = $0-$83,466,000

#### **Clustering Results:**
- **Cluster 0** (High-risk): 1,117 customers - High recency (46.7 days), Low frequency (2.2), Low monetary ($2,997)
- **Cluster 1** (Medium-risk): 1,633 customers - Moderate recency (36.9 days), Frequency (11.2), Monetary ($128,212)
- **Cluster 2** (Low-risk): 992 customers - Low recency (5.5 days), High frequency (75.6), High monetary ($584,756)

#### **Target Distribution:**

# Customer Level:
- High-risk customers: 1,117 (29.85%)
- Low-risk customers: 2,625 (70.15%)

# Transaction Level:
- High-risk transactions: 2,399 (2.51%)
- Low-risk transactions: 93,263 (97.49%)


#### **Key Design Decisions:**
1. **Snapshot Date**: Used max transaction date + 1 day for consistent recency calculation
2. **Scaling Strategy**: Log transformation followed by StandardScaler before K-Means
3. **Risk Identification**: Weighted combination of normalized RFM metrics (40% recency, 40% frequency, 20% monetary)
4. **Data Integration**: Robust handling of CustomerId format mismatches between raw and engineered data

### 📈 Business Insights

1. **Clear Customer Segmentation**: RFM clustering reveals three distinct customer tiers
2. **High-Risk Characteristics**: Customers with >46 days since last purchase, <3 transactions, <$3,000 spending
3. **Risk Proportion**: ~30% of customers identified as high-risk, but only ~2.5% of transactions
4. **Model Readiness**: Target variable created with reasonable class distribution for credit risk modeling

### 💾 Deliverables Produced

1. **Source Code**: `src/target_engineering.py` - Complete OOP implementation
2. **Demo Notebook**: `notebooks/task4_target_engineering.ipynb` - Step-by-step execution
3. **Processed Data**:
   - `data/processed/task4_target_engineered.parquet` - Main dataset with target (95,662×41)
   - `data/processed/task4_customer_risk_mapping.csv` - Customer-risk mapping (3,742×2)
4. **Visualizations**: RFM distributions, cluster plots, risk analysis charts

### 🔄 Integration with Previous Tasks

- **Input**: Raw transaction data + Task 3 engineered features
- **Process**: RFM calculation → Clustering → Risk labeling → Integration
- **Output**: Task 3 features augmented with `is_high_risk` target variable
- **Ready for**: Task 5 model training with complete feature-target dataset

### ⚠️ Challenges & Solutions

1. **CustomerId Format Mismatch**: Raw data vs Task 3 engineered features had different CustomerId formats
   - **Solution**: Robust handling with index resetting and format normalization
2. **High Skewness in Monetary Values**: Extreme value ranges affected clustering
   - **Solution**: Log transformation before scaling
3. **Transaction vs Customer Level Proportions**: Different risk proportions at different levels
   - **Solution**: Both levels analyzed and documented

### 🎯 Validation & Quality Checks

 **RFM Calculation**: All 3,742 customers have RFM metrics  
 **Clustering Quality**: Silhouette Score > 0.3 indicates meaningful clusters  
 **Risk Identification**: High-risk cluster aligns with business logic (disengaged customers)  
**Data Integration**: All transactions have risk labels after merge  
 **Reproducibility**: Fixed random seeds ensure consistent results

## Task 5: Model Training and Tracking

### Objective
Develop a structured model training process with experiment tracking, model versioning, and unit testing to build a credit risk prediction model.

### Methodology
1. **Data Preparation**:
   - Loaded engineered features from Task 3-4
   - Removed potential data leakage (`FraudResult` column)
   - Applied SMOTE to handle severe class imbalance (2.5% positive class)
   - Split data: 80% training, 20% testing with stratification

2. **Model Selection**:
   - Logistic Regression (baseline with interpretability)
   - Random Forest (ensemble method for complex patterns)
   - XGBoost (gradient boosting for high performance)

3. **Hyperparameter Tuning**:
   - Used RandomizedSearchCV for efficient search
   - Tuned key parameters for each algorithm
   - Evaluated using ROC-AUC as primary metric

4. **Experiment Tracking**:
   - Implemented MLflow for experiment management
   - Tracked parameters, metrics, and artifacts
   - Used SQLite backend for local tracking

5. **Evaluation Metrics**:
   - ROC-AUC Score
   - F1-Score (important for imbalanced data)
   - Precision and Recall
   - Accuracy and Specificity
   - Confusion Matrix analysis

### Key Findings
1. **Data Leakage Prevention**: Identified and removed `FraudResult` column which was artificially inflating model performance.

2. **Class Imbalance Challenge**: Original dataset had only 2.5% high-risk customers. Applied SMOTE to create balanced training data.

3. **Model Performance**:
   - **Logistic Regression**: ROC-AUC = 0.9902, F1 = 0.4503
   - **Random Forest**: ROC-AUC = 1.0000, F1 = 0.9846
   - **XGBoost**: ROC-AUC = 1.0000, F1 = 0.9776

4. **Overfitting Detection**: Random Forest and XGBoost show perfect ROC-AUC scores, indicating potential overfitting despite SMOTE application.

5. **Production Model Selection**: Selected **Logistic Regression** for deployment due to:
   - Realistic performance metrics (not perfect scores)
   - Better interpretability for financial context
   - Lower risk of overfitting on synthetic data

### Artifacts Generated
- **Models**: `models/logistic_regression_model.pkl`, `models/random_forest_model.pkl`, `models/xgboost_model.pkl`
- **Preprocessing**: `models/scaler.pkl`, `models/feature_names.pkl`
- **Evaluation**: `reports/model_comparison.csv`, `reports/metrics.json`
- **Visualizations**: ROC curves and confusion matrices in `reports/plots/`
- **MLflow Database**: `mlflow.db` with complete experiment history

### Unit Tests
- Created comprehensive unit tests in `tests/test_data_processing.py`
- Tested data validation, feature encoding, and model evaluation logic
- Ensured reproducibility with fixed random seeds

### Limitations and Considerations
1. **SMOTE Limitations**: Synthetic samples may create unrealistic patterns that models can memorize
2. **Time-Based Validation**: Future work should implement time-based splits instead of random splits
3. **Feature Importance**: Need deeper analysis of which features drive predictions
4. **Calibration**: Probability scores may need calibration for production use

