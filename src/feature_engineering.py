"""
Feature Engineering OOP Classes
Contains all 6 required components for Task 3
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 1. CREATE AGGREGATE FEATURES
# ============================================
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Step 1: Create customer-level aggregate features
    Requirements:
    - Total Transaction Amount per customer
    - Average Transaction Amount per customer  
    - Transaction Count per customer
    - Standard Deviation of Transaction Amounts per customer
    """
    
    def __init__(self, customer_col='CustomerId', amount_col='Amount'):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.aggregate_features = None
        
    def fit(self, X, y=None):
        """Calculate aggregate statistics per customer"""
        if self.customer_col in X.columns and self.amount_col in X.columns:
            # Create customer aggregates
            self.aggregate_features = X.groupby(self.customer_col)[self.amount_col].agg([
                ('TotalAmount', 'sum'),
                ('AvgAmount', 'mean'),
                ('TransactionCount', 'count'),
                ('StdAmount', 'std'),
                ('MinAmount', 'min'),
                ('MaxAmount', 'max'),
                ('MedianAmount', 'median')
            ]).reset_index()
        return self
    
    def transform(self, X):
        """Merge aggregate features back to transaction data"""
        X = X.copy()
        
        if self.aggregate_features is not None:
            # Merge customer aggregates
            X = pd.merge(
                X, 
                self.aggregate_features, 
                on=self.customer_col, 
                how='left',
                suffixes=('', '_Customer')
            )
            
            print(f"✅ Step 1: Added {len(self.aggregate_features.columns)-1} aggregate features")
            
        return X


# ============================================
# 2. EXTRACT TEMPORAL FEATURES  
# ============================================
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Step 2: Extract temporal features from TransactionStartTime
    Requirements:
    - Transaction Hour
    - Transaction Day
    - Transaction Month  
    - Transaction Year
    """
    
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract temporal features"""
        X = X.copy()
        
        if self.datetime_col in X.columns:
            # Ensure datetime format
            if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
                X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
            
            # Extract basic temporal features
            X['TransactionHour'] = X[self.datetime_col].dt.hour
            X['TransactionDay'] = X[self.datetime_col].dt.day
            X['TransactionMonth'] = X[self.datetime_col].dt.month
            X['TransactionYear'] = X[self.datetime_col].dt.year
            
            # Additional useful features
            X['TransactionDayOfWeek'] = X[self.datetime_col].dt.dayofweek
            X['TransactionWeekOfYear'] = X[self.datetime_col].dt.isocalendar().week
            X['IsWeekend'] = (X['TransactionDayOfWeek'] >= 5).astype(int)
            
            print(f"✅ Step 2: Added 7 temporal features")
            
        return X


# ============================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Step 3: Encode categorical variables
    Requirements:
    - One-Hot Encoding: Converts categorical values into binary vectors
    - Label Encoding: Assigns a unique integer to each category
    """
    
    def __init__(self, strategy='onehot', columns=None):
        """
        Parameters:
        -----------
        strategy : str ('onehot' or 'label')
            Encoding strategy to use
        columns : list
            Specific columns to encode (None = auto-detect)
        """
        self.strategy = strategy
        self.columns = columns
        self.encoders = {}
        self.encoded_columns = []
        
    def fit(self, X, y=None):
        """Fit encoders to categorical columns"""
        if self.columns is None:
            # Auto-detect categorical columns
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                if self.strategy == 'onehot':
                    # OneHotEncoder stores categories for transform
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoder.fit(X[[col]])
                    self.encoders[col] = encoder
                    
                    # Store column names for transformation
                    categories = encoder.categories_[0]
                    self.encoded_columns.extend([f"{col}_{cat}" for cat in categories])
                    
                elif self.strategy == 'label':
                    # Simple label encoding
                    unique_vals = X[col].dropna().unique()
                    self.encoders[col] = {val: idx for idx, val in enumerate(unique_vals)}
        
        return self
    
    def transform(self, X):
        """Transform categorical columns"""
        X = X.copy()
        
        for col, encoder in self.encoders.items():
            if col in X.columns:
                if self.strategy == 'onehot':
                    # Apply one-hot encoding
                    encoded = encoder.transform(X[[col]])
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                        index=X.index
                    )
                    
                    # Drop original column and add encoded ones
                    X = X.drop(columns=[col])
                    X = pd.concat([X, encoded_df], axis=1)
                    
                elif self.strategy == 'label':
                    # Apply label encoding
                    X[col] = X[col].map(encoder).fillna(-1)
        
        print(f"✅ Step 3: Encoded {len(self.encoders)} categorical columns using {self.strategy} encoding")
        return X


# ============================================
# 4. HANDLE MISSING VALUES
# ============================================
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Step 4: Handle missing values
    Requirements:
    - Imputation: Fill missing values with mean, median, mode, or KNN
    - Removal: Remove rows or columns with missing values
    """
    
    def __init__(self, strategy='median', remove_threshold=0.5):
        """
        Parameters:
        -----------
        strategy : str ('mean', 'median', 'mode', 'knn')
            Imputation strategy
        remove_threshold : float
            Remove columns with more than this % missing
        """
        self.strategy = strategy
        self.remove_threshold = remove_threshold
        self.imputers = {}
        self.columns_to_keep = []
        
    def fit(self, X, y=None):
        """Analyze missing values and prepare imputers"""
        # Calculate missing percentages
        missing_pct = X.isnull().sum() / len(X)
        
        # Identify columns to remove
        self.columns_to_keep = missing_pct[missing_pct < self.remove_threshold].index.tolist()
        
        # Prepare imputers for remaining columns with missing values
        for col in self.columns_to_keep:
            if X[col].isnull().any():
                if self.strategy == 'knn':
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    # KNN needs numeric data
                    if pd.api.types.is_numeric_dtype(X[col]):
                        self.imputers[col] = imputer
                else:
                    imputer = SimpleImputer(strategy=self.strategy)
                    self.imputers[col] = imputer
        
        return self
    
    def transform(self, X):
        """Handle missing values"""
        X = X.copy()
        
        # Remove high-missing columns
        if len(self.columns_to_keep) < len(X.columns):
            X = X[self.columns_to_keep]
            print(f"⚠️  Step 4: Removed {len(X.columns) - len(self.columns_to_keep)} columns with >{self.remove_threshold*100}% missing")
        
        # Impute missing values
        for col, imputer in self.imputers.items():
            if col in X.columns:
                if isinstance(imputer, SimpleImputer):
                    X[col] = imputer.fit_transform(X[[col]]).ravel()
                elif str(type(imputer)).find('KNNImputer') != -1:
                    X[col] = imputer.fit_transform(X[[col]]).ravel()
        
        print(f"✅ Step 4: Handled missing values using {self.strategy} strategy")
        return X


# ============================================
# 5. NORMALIZE/STANDARDIZE NUMERICAL FEATURES
# ============================================
class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Step 5: Normalize/Standardize numerical features
    Requirements:
    - Normalization: Scale data to range [0, 1]
    - Standardization: Scale data to mean=0, std=1
    """
    
    def __init__(self, strategy='standard', columns=None):
        """
        Parameters:
        -----------
        strategy : str ('standard', 'minmax', 'robust')
            Scaling strategy
        columns : list
            Columns to scale (None = auto-detect numeric)
        """
        self.strategy = strategy
        self.columns = columns
        self.scalers = {}
        
    def fit(self, X, y=None):
        """Fit scalers to numerical columns"""
        if self.columns is None:
            # Auto-detect numeric columns
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                if self.strategy == 'standard':
                    scaler = StandardScaler()
                elif self.strategy == 'minmax':
                    scaler = MinMaxScaler()
                elif self.strategy == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()
                
                # Fit scaler
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
        
        return self
    
    def transform(self, X):
        """Scale numerical features"""
        X = X.copy()
        
        for col, scaler in self.scalers.items():
            if col in X.columns:
                X[col] = scaler.transform(X[[col]]).ravel()
        
        print(f"✅ Step 5: Scaled {len(self.scalers)} numerical features using {self.strategy}")
        return X


# ============================================
# 6. FEATURE ENGINEERING WITH WOE AND IV (FIXED)
# ============================================
class WOETransformer(BaseEstimator, TransformerMixin):
    """
    Step 6: Weight of Evidence (WoE) and Information Value (IV)
    Requirements:
    - Calculate WoE for categorical/binned features
    - Calculate IV to measure predictive power
    """
    
    def __init__(self, n_bins=5):
        """
        Parameters:
        -----------
        n_bins : int
            Number of bins for numerical features
        """
        self.n_bins = n_bins
        self.woe_dict = {}
        self.iv_dict = {}
        
    def fit(self, X, y):
        """Calculate WoE and IV for features"""
        print(f"🔍 Step 6: Calculating WoE/IV for {X.shape[1]} features...")
        
        # Validate inputs
        if y is None:
            raise ValueError("Target variable 'y' is required for WoE calculation")
        
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)}")
        
        # Calculate for each feature
        for col in X.columns:
            if X[col].nunique() > 1:  # Skip constant features
                self._calculate_feature_woe(X[col], y, col)
        
        print(f"✅ WoE calculation complete for {len(self.woe_dict)} features")
        return self
    
    def _calculate_feature_woe(self, feature, target, col_name):
        """Calculate WoE for a single feature"""
        try:
            # For numerical features with many values, bin first
            if pd.api.types.is_numeric_dtype(feature) and feature.nunique() > self.n_bins:
                try:
                    feature_binned = pd.qcut(feature, q=self.n_bins, duplicates='drop', labels=False)
                    feature_binned = feature_binned.astype(str) + "_bin"
                except:
                    # If qcut fails, use equal width binning
                    feature_binned = pd.cut(feature, bins=self.n_bins, labels=False)
                    feature_binned = feature_binned.astype(str) + "_bin"
            else:
                feature_binned = feature.astype(str)
            
            # Create calculation dataframe
            woe_df = pd.DataFrame({
                'feature': feature_binned.fillna('MISSING'),
                'target': target
            })
            
            # Group by feature value
            grouped = woe_df.groupby('feature')['target'].agg(['sum', 'count'])
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']
            
            # Add smoothing (add 0.5 to avoid division by zero)
            grouped['bad'] += 0.5
            grouped['good'] += 0.5
            
            # Calculate percentages
            total_good = grouped['good'].sum()
            total_bad = grouped['bad'].sum()
            
            grouped['pct_good'] = grouped['good'] / total_good
            grouped['pct_bad'] = grouped['bad'] / total_bad
            
            # Calculate WoE
            grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
            
            # Calculate IV component
            grouped['iv_component'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']
            
            # Store results
            self.woe_dict[col_name] = grouped['woe'].to_dict()
            self.iv_dict[col_name] = grouped['iv_component'].sum()
            
        except Exception as e:
            print(f"⚠️  Could not calculate WoE for {col_name}: {str(e)}")
            self.woe_dict[col_name] = {}
            self.iv_dict[col_name] = 0
    
    def transform(self, X):
        """Transform features using WoE values"""
        X_transformed = X.copy()
        
        for col, woe_mapping in self.woe_dict.items():
            if col in X.columns and woe_mapping:
                # Apply WoE transformation
                if pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > self.n_bins:
                    # Bin numerical features (same as during fit)
                    try:
                        feature_binned = pd.qcut(X[col], q=self.n_bins, duplicates='drop', labels=False)
                        feature_binned = feature_binned.astype(str) + "_bin"
                        X_transformed[f'{col}_WOE'] = feature_binned.map(woe_mapping)
                    except:
                        # Fallback binning
                        feature_binned = pd.cut(X[col], bins=self.n_bins, labels=False)
                        feature_binned = feature_binned.astype(str) + "_bin"
                        X_transformed[f'{col}_WOE'] = feature_binned.map(woe_mapping)
                else:
                    # Direct mapping for categorical
                    X_transformed[f'{col}_WOE'] = X[col].astype(str).map(woe_mapping)
                
                # Fill missing WoE with neutral value (0)
                X_transformed[f'{col}_WOE'].fillna(0, inplace=True)
        
        print(f"✅ Step 6: Created {len(self.woe_dict)} WoE features")
        return X_transformed

# ============================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================
class FeatureEngineeringPipeline:
    """Orchestrates all 6 feature engineering steps"""
    
    def __init__(self):
        self.pipeline = None
        self.steps = []
        
    def add_step(self, name, transformer):
        """Add a step to the pipeline"""
        self.steps.append((name, transformer))
    
    def build_pipeline(self):
        """Build the complete sklearn pipeline"""
        from sklearn.pipeline import Pipeline
        
        self.pipeline = Pipeline(self.steps)
        return self.pipeline
    
    def create_default_pipeline(self):
        """Create default pipeline with all 6 steps"""
        self.steps = [
            ('aggregate_features', AggregateFeatures()),
            ('temporal_features', TemporalFeatureExtractor()),
            ('missing_values', MissingValueHandler(strategy='median')),
            ('categorical_encoding', CategoricalEncoder(strategy='onehot')),
            ('feature_scaling', FeatureScaler(strategy='standard')),
            # WoE requires target, will be added separately
        ]
        
        self.pipeline = Pipeline(self.steps)
        return self.pipeline