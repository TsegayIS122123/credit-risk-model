"""
Data Preparation Class for Task 5
Handles data loading, cleaning, splitting, and preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreparation:
    """Class for preparing data for model training"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_customer_level = False
        
    def load_data(self):
        """Load and validate data"""
        print("📥 Loading data...")
        
        data_path = self.config.find_data_file()
        
        # Load based on file extension
        if data_path.suffix == '.parquet':
            self.data = pd.read_parquet(data_path)
        else:
            self.data = pd.read_csv(data_path)
        
        print(f"   Loaded {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def validate_and_clean(self):
        """Validate data and perform cleaning"""
        print("🧹 Validating and cleaning data...")
        
        # Check for target column
        if self.config.TARGET_COL not in self.data.columns:
            raise ValueError(f"Target column '{self.config.TARGET_COL}' not found in data")
        
        # Check data level (customer vs transaction)
        if 'CustomerId' in self.data.columns:
            unique_customers = self.data['CustomerId'].nunique()
            total_rows = len(self.data)
            
            if unique_customers < total_rows:
                print(f"   ⚠️  Data is transaction-level ({total_rows} transactions, {unique_customers} customers)")
                print("   🔄 Converting to customer-level...")
                self._convert_to_customer_level()
                self.is_customer_level = True
            else:
                print(f"   ✅ Data is customer-level ({unique_customers} customers)")
                self.is_customer_level = True
        
        # Remove ID columns
        id_cols_to_remove = [col for col in self.config.ID_COLS if col in self.data.columns]
        self.data = self.data.drop(columns=id_cols_to_remove, errors='ignore')
        
        # Handle missing values in target
        if self.data[self.config.TARGET_COL].isnull().any():
            print(f"   ⚠️  Found {self.data[self.config.TARGET_COL].isnull().sum()} missing values in target")
            self.data = self.data.dropna(subset=[self.config.TARGET_COL])
        
        print(f"   After cleaning: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def _convert_to_customer_level(self):
        """Convert transaction data to customer-level"""
        # Use RFM features if available
        rfm_features = ['Recency', 'Frequency', 'Monetary', 'Cluster', 'is_high_risk']
        available_rfm = [f for f in rfm_features if f in self.data.columns]
        
        if available_rfm and 'CustomerId' in self.data.columns:
            # Use existing RFM features
            customer_cols = ['CustomerId'] + available_rfm
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Add additional numeric features
            additional_features = [col for col in numeric_cols 
                                 if col not in customer_cols and col != self.config.TARGET_COL]
            
            if additional_features:
                # Aggregate additional features
                agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in additional_features[:10]}  # Limit to 10
                
                # Group by customer
                grouped = self.data.groupby('CustomerId').agg({
                    **{col: 'first' for col in available_rfm if col != 'CustomerId'},
                    **agg_dict
                })
                
                # Flatten multi-index columns
                grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
                self.data = grouped.reset_index()
        else:
            # Fallback: simple aggregation
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'CustomerId']
            
            if numeric_cols:
                agg_dict = {col: 'mean' for col in numeric_cols[:20]}  # Limit features
                self.data = self.data.groupby('CustomerId').agg(agg_dict).reset_index()
    
    def split_data(self):
        """Split data into train and test sets"""
        print("✂️  Splitting data into train/test sets...")
        
        # Separate features and target
        feature_cols = [col for col in self.data.columns 
                       if col != self.config.TARGET_COL]
        
        X = self.data[feature_cols]
        y = self.data[self.config.TARGET_COL]
        
        # Handle categorical features
        X = self._encode_categorical_features(X)
        
        # Handle missing values in features
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        self.feature_names = X.columns.tolist()
        
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        print(f"   Target distribution - Train: {self.y_train.value_counts().to_dict()}")
        print(f"   Target distribution - Test: {self.y_test.value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            
            if unique_count <= 2:
                # Binary encoding
                X[col] = X[col].astype('category').cat.codes
            elif unique_count <= self.config.MAX_CATEGORICAL_UNIQUE:
                # Label encoding for low cardinality
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            else:
                # Drop high cardinality columns
                print(f"   Dropping high cardinality column: {col} ({unique_count} unique values)")
                X = X.drop(columns=[col])
        
        return X
    
    def scale_features(self):
        """Scale features using StandardScaler"""
        print("📏 Scaling features...")
        
        # Fit on training, transform both
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
        print("   Features scaled using StandardScaler")
        return self.X_train, self.X_test
    
    def get_data_summary(self):
        """Get summary of prepared data"""
        return {
            "data_level": "customer" if self.is_customer_level else "transaction",
            "original_shape": self.data.shape,
            "train_shape": self.X_train.shape,
            "test_shape": self.X_test.shape,
            "n_features": len(self.feature_names),
            "train_class_dist": self.y_train.value_counts().to_dict(),
            "test_class_dist": self.y_test.value_counts().to_dict(),
            "imbalance_ratio": self.y_train.value_counts().get(0, 1) / 
                              max(self.y_train.value_counts().get(1, 1), 1)
        }