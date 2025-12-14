"""
Task 4 - Proxy Target Variable Engineering
Classes for creating credit risk proxy target using RFM analysis and clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CLASS 1: RFM METRICS CALCULATOR
# ============================================
class RFMMetricsCalculator(BaseEstimator, TransformerMixin):
    """
    Instruction 1: Calculate RFM Metrics
    
    For each CustomerId, calculate their Recency, Frequency, and Monetary (RFM) 
    values from the transaction history.
    
    Define a snapshot date to calculate Recency consistently.
    """
    
    def __init__(self, 
                 customer_col='CustomerId',
                 datetime_col='TransactionStartTime',
                 amount_col='Amount',
                 snapshot_date=None):
        """
        Parameters:
        -----------
        customer_col : str, default='CustomerId'
            Column containing customer identifiers
        datetime_col : str, default='TransactionStartTime'
            Column containing transaction timestamps
        amount_col : str, default='Amount'
            Column containing transaction amounts
        snapshot_date : pd.Timestamp, optional
            Reference date for recency calculation. If None, uses max date + 1 day
        """
        self.customer_col = customer_col
        self.datetime_col = datetime_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date
        self.rfm_data = None
        self.rfm_summary = None
        
    def fit(self, X, y=None):
        """Calculate RFM metrics from transaction data"""
        return self
    
    def transform(self, X):
        """
        Calculate RFM metrics:
        - Recency: Days since last transaction (from snapshot date)
        - Frequency: Number of transactions
        - Monetary: Total amount spent (absolute value)
        """
        X = X.copy()
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        # Define snapshot date (Instruction: "Define a snapshot date consistently")
        if self.snapshot_date is None:
            self.snapshot_date = X[self.datetime_col].max() + pd.Timedelta(days=1)
        print(f"📅 Snapshot date for recency calculation: {self.snapshot_date}")
        
        # FIX: Check if TransactionId exists, otherwise create a dummy counter
        if 'TransactionId' in X.columns:
            frequency_counter = 'TransactionId'
        else:
            # Create a dummy column for counting transactions
            X['_transaction_counter'] = 1
            frequency_counter = '_transaction_counter'
            print("⚠️  TransactionId column not found, using row count for frequency")
        
        # FIXED: Calculate RFM metrics properly - use only positive amounts for Monetary
        aggregation_dict = {
            self.datetime_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency
            frequency_counter: 'count',  # Frequency
            self.amount_col: lambda x: x[x > 0].sum()  # FIXED: Only positive amounts (spending)
        }
        
        rfm = X.groupby(self.customer_col).agg(aggregation_dict).rename(columns={
            self.datetime_col: 'Recency',
            frequency_counter: 'Frequency',
            self.amount_col: 'Monetary'
        })
        
        # Clean up dummy column if created
        if '_transaction_counter' in X.columns:
            X.drop('_transaction_counter', axis=1, inplace=True)
        
        # Handle customers with no positive transactions
        rfm['Monetary'] = rfm['Monetary'].fillna(0)
        
        # FIXED: Apply log transformation for clustering (as per instruction)
        rfm['Recency_Log'] = np.log1p(rfm['Recency'])
        rfm['Frequency_Log'] = np.log1p(rfm['Frequency'])
        rfm['Monetary_Log'] = np.log1p(rfm['Monetary'] + 1)  # +1 to handle zeros
        
        # Remove the customer_stats calculation to keep it simple
        self.rfm_data = rfm
        
        # Summary statistics
        self.rfm_summary = self.rfm_data[['Recency', 'Frequency', 'Monetary']].describe()
        
        print(f"✅ RFM metrics calculated for {len(self.rfm_data)} unique customers")
        print(f"📊 RFM Summary:\n{self.rfm_summary}")
        
        return self.rfm_data
    
    def plot_rfm_distributions(self):
        """Visualize RFM distributions"""
        if self.rfm_data is None:
            raise ValueError("Run transform() first to calculate RFM metrics")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original RFM distributions
        axes[0, 0].hist(self.rfm_data['Recency'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days since last transaction')
        
        axes[0, 1].hist(self.rfm_data['Frequency'], bins=50, edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of transactions')
        
        axes[0, 2].hist(self.rfm_data['Monetary'], bins=50, edgecolor='black')
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total amount spent')
        
        # Log-transformed distributions
        axes[1, 0].hist(self.rfm_data['Recency_Log'], bins=50, edgecolor='black')
        axes[1, 0].set_title('Log(Recency) Distribution')
        
        axes[1, 1].hist(self.rfm_data['Frequency_Log'], bins=50, edgecolor='black')
        axes[1, 1].set_title('Log(Frequency) Distribution')
        
        axes[1, 2].hist(self.rfm_data['Monetary_Log'], bins=50, edgecolor='black')
        axes[1, 2].set_title('Log(Monetary) Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    


# ============================================
# CLASS 2: RFM CLUSTERING
# ============================================
class RFMClustering(BaseEstimator, TransformerMixin):
    """
    Instruction 2: Cluster Customers
    
    Use the K-Means clustering algorithm to segment customers into 3 distinct 
    groups based on their RFM profiles.
    
    Pre-process (e.g., scale) the RFM features appropriately before clustering 
    to ensure that the results are meaningful.
    
    Set a random_state during clustering to ensure reproducibility.
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters for K-Means (Instruction: "segment into 3 distinct groups")
        random_state : int, default=42
            Random state for reproducibility (Instruction: "Set a random_state")
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = None
        self.cluster_centers = None
        self.silhouette_score = None
        
    def fit(self, X, y=None):
        """Fit K-Means clustering on RFM data"""
        # Instruction: "Pre-process (e.g., scale) the RFM features appropriately"
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # FIXED: Use log-transformed features for scaling
        features_to_scale = ['Recency_Log', 'Frequency_Log', 'Monetary_Log']
        
        # Check if log features exist, if not create them
        for feature in ['Recency_Log', 'Frequency_Log', 'Monetary_Log']:
            if feature not in X.columns:
                if 'Recency_Log' not in X.columns:
                    X['Recency_Log'] = np.log1p(X['Recency'])
                if 'Frequency_Log' not in X.columns:
                    X['Frequency_Log'] = np.log1p(X['Frequency'])
                if 'Monetary_Log' not in X.columns:
                    X['Monetary_Log'] = np.log1p(X['Monetary'] + 1)
                break
        
        # Scale the features (THIS WAS MISSING - per instruction)
        X_scaled = self.scaler.fit_transform(X[features_to_scale])
        
        # Instruction: "Use the K-Means clustering algorithm"
        self.kmeans.fit(X_scaled)
        self.cluster_labels = self.kmeans.labels_
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate silhouette score for clustering quality
        self.silhouette_score = silhouette_score(X_scaled, self.cluster_labels)
        
        print(f"✅ K-Means clustering complete with {self.n_clusters} clusters")
        print(f"📊 Silhouette Score: {self.silhouette_score:.4f}")
        print(f"📈 Cluster sizes: {pd.Series(self.cluster_labels).value_counts().sort_index().to_dict()}")
        
        return self
    
    def transform(self, X):
        """Assign cluster labels to RFM data"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Ensure we have the same features used during fit
        X_scaled = self.scaler.transform(X[self.features_to_scale])
        
        # Predict clusters
        clusters = self.kmeans.predict(X_scaled)
        
        # Add cluster labels to data
        X_with_clusters = X.copy()
        X_with_clusters['Cluster'] = clusters
        
        # Reset index if CustomerId is in index
        if X_with_clusters.index.name == 'CustomerId':
            X_with_clusters = X_with_clusters.reset_index()
        
        return X_with_clusters
    
    def plot_clusters(self, X):
        """Visualize RFM clusters in 2D and 3D"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_scaled = self.scaler.transform(X[self.features_to_scale])
        
        fig = plt.figure(figsize=(18, 6))
        
        # 2D Plots
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                             c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Recency (scaled)')
        plt.ylabel('Frequency (scaled)')
        plt.title('RFM Clusters: Recency vs Frequency')
        plt.colorbar(scatter)
        
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(X_scaled[:, 1], X_scaled[:, 2], 
                             c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Frequency (scaled)')
        plt.ylabel('Monetary (scaled)')
        plt.title('RFM Clusters: Frequency vs Monetary')
        plt.colorbar(scatter)
        
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 2], 
                             c=self.cluster_labels, cmap='viridis', alpha=0.6)
        plt.xlabel('Recency (scaled)')
        plt.ylabel('Monetary (scaled)')
        plt.title('RFM Clusters: Recency vs Monetary')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def get_cluster_profiles(self, X):
        """Analyze and describe each cluster's characteristics"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_with_clusters = self.transform(X)
        
        # Calculate cluster statistics
        cluster_profiles = X_with_clusters.groupby('Cluster').agg({
            'Recency': ['mean', 'std', 'min', 'max'],
            'Frequency': ['mean', 'std', 'min', 'max'],
            'Monetary': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        # Add interpretation
        interpretations = []
        for cluster_id in range(self.n_clusters):
            cluster_data = X_with_clusters[X_with_clusters['Cluster'] == cluster_id]
            
            # Determine cluster characteristics
            recency_mean = cluster_data['Recency'].mean()
            frequency_mean = cluster_data['Frequency'].mean()
            monetary_mean = cluster_data['Monetary'].mean()
            
            if recency_mean > X['Recency'].mean() and frequency_mean < X['Frequency'].mean():
                interpretation = "High-Risk (Disengaged)"
            elif frequency_mean > X['Frequency'].mean() and monetary_mean > X['Monetary'].mean():
                interpretation = "Low-Risk (Loyal Customers)"
            else:
                interpretation = "Medium-Risk (Needs Attention)"
            
            interpretations.append(interpretation)
        
        cluster_profiles['Interpretation'] = interpretations
        
        print("📋 Cluster Profiles:")
        print(cluster_profiles)
        
        return cluster_profiles


# ============================================
# CLASS 3: HIGH-RISK LABEL ASSIGNER
# ============================================
class HighRiskLabelAssigner(BaseEstimator, TransformerMixin):
    """
    Instruction 3: Define and Assign the "High-Risk" Label
    
    Analyze the resulting clusters to determine which one represents the least 
    engaged (highest-risk) customer segment (typically characterized by low 
    frequency and low monetary value).
    
    Create a new binary target column named is_high_risk.
    Assign a value of 1 to customers in the high-risk cluster and 0 to all others.
    """
    
    def __init__(self, risk_strategy='rfm_weighted'):
        """
        Parameters:
        -----------
        risk_strategy : str
            Strategy to identify high-risk cluster:
            - 'rfm_weighted': Weighted combination of RFM (recommended)
            - 'lowest_frequency': Cluster with lowest average frequency
            - 'highest_recency': Cluster with highest average recency
        """
        self.risk_strategy = risk_strategy
        self.high_risk_cluster = None
        self.cluster_analysis = None
        
    def fit(self, X_with_clusters, rfm_data):
        """
        Identify which cluster represents high-risk customers
        
        Parameters:
        -----------
        X_with_clusters : pd.DataFrame
            RFM data with cluster labels
        rfm_data : pd.DataFrame
            Original RFM data for comparison
        """
        if 'Cluster' not in X_with_clusters.columns:
            raise ValueError("Input data must contain 'Cluster' column")
        
        # Reset index if CustomerId is in index
        if X_with_clusters.index.name == 'CustomerId':
            X_with_clusters = X_with_clusters.reset_index()
        
        # Calculate cluster statistics
        self.cluster_analysis = X_with_clusters.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean', 
            'Monetary': 'mean'
        })
        
        # FIXED: Better risk identification logic
        if self.risk_strategy == 'rfm_weighted':
            # Get overall means for normalization
            recency_mean = X_with_clusters['Recency'].mean()
            frequency_mean = X_with_clusters['Frequency'].mean()
            monetary_mean = X_with_clusters['Monetary'].mean()
            
            risk_scores = []
            for cluster_id in sorted(X_with_clusters['Cluster'].unique()):
                cluster_data = X_with_clusters[X_with_clusters['Cluster'] == cluster_id]
                
                # Risk factors (higher = more risk):
                # 1. Higher than average recency (inactive longer) = more risk
                # 2. Lower than average frequency (buys less often) = more risk
                # 3. Lower than average monetary (spends less) = more risk
                
                recency_risk = cluster_data['Recency'].mean() / recency_mean
                frequency_risk = 1 - (cluster_data['Frequency'].mean() / frequency_mean)
                monetary_risk = 1 - (cluster_data['Monetary'].mean() / monetary_mean)
                
                # Weighted risk score (business decision - prioritize frequency)
                total_risk = recency_risk * 0.4 + frequency_risk * 0.4 + monetary_risk * 0.2
                risk_scores.append(total_risk)
                
                print(f"   Cluster {cluster_id}: Risk Score = {total_risk:.3f} "
                      f"(R={cluster_data['Recency'].mean():.1f} days, "
                      f"F={cluster_data['Frequency'].mean():.1f} trans, "
                      f"M=${cluster_data['Monetary'].mean():.0f})")
            
            self.high_risk_cluster = np.argmax(risk_scores)
            
        elif self.risk_strategy == 'lowest_frequency':
            # Cluster with lowest average frequency
            self.high_risk_cluster = self.cluster_analysis['Frequency'].idxmin()
            
        elif self.risk_strategy == 'highest_recency':
            # Cluster with highest average recency
            self.high_risk_cluster = self.cluster_analysis['Recency'].idxmax()
        
        print(f"\n🎯 High-risk cluster identified: Cluster {self.high_risk_cluster}")
        print(f"📊 High-risk cluster characteristics:")
        print(f"   Recency: {self.cluster_analysis.loc[self.high_risk_cluster, 'Recency']:.1f} days")
        print(f"   Frequency: {self.cluster_analysis.loc[self.high_risk_cluster, 'Frequency']:.1f} transactions")
        print(f"   Monetary: ${self.cluster_analysis.loc[self.high_risk_cluster, 'Monetary']:.2f}")
        
        return self
    
    def transform(self, X_with_clusters):
        """Assign high-risk labels to customers"""
        # Instruction: "Create a new binary target column named is_high_risk"
        X_labeled = X_with_clusters.copy()
        
        # Reset index if CustomerId is in index
        if X_labeled.index.name == 'CustomerId':
            X_labeled = X_labeled.reset_index()
        
        # Instruction: "Assign a value of 1 to customers in the high-risk cluster and 0 to all others"
        X_labeled['is_high_risk'] = (X_labeled['Cluster'] == self.high_risk_cluster).astype(int)
        
        risk_stats = X_labeled['is_high_risk'].value_counts()
        print(f"✅ High-risk labels assigned: {risk_stats[1]} high-risk vs {risk_stats[0]} low-risk customers")
        print(f"📈 High-risk proportion: {(risk_stats[1]/len(X_labeled)*100):.2f}%")
        
        return X_labeled
    
    def plot_risk_distribution(self, X_labeled):
        """Visualize risk distribution across clusters"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Box plots by cluster
        axes[0].boxplot([X_labeled[X_labeled['Cluster'] == i]['Recency'] 
                        for i in sorted(X_labeled['Cluster'].unique())])
        axes[0].set_title('Recency by Cluster')
        axes[0].set_xlabel('Cluster')
        axes[0].set_ylabel('Recency (days)')
        axes[0].axvline(x=self.high_risk_cluster + 1, color='r', linestyle='--', 
                       label=f'High-risk (Cluster {self.high_risk_cluster})')
        
        axes[1].boxplot([X_labeled[X_labeled['Cluster'] == i]['Frequency'] 
                        for i in sorted(X_labeled['Cluster'].unique())])
        axes[1].set_title('Frequency by Cluster')
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(x=self.high_risk_cluster + 1, color='r', linestyle='--')
        
        axes[2].boxplot([X_labeled[X_labeled['Cluster'] == i]['Monetary'] 
                        for i in sorted(X_labeled['Cluster'].unique())])
        axes[2].set_title('Monetary by Cluster')
        axes[2].set_xlabel('Cluster')
        axes[2].set_ylabel('Monetary')
        axes[2].axvline(x=self.high_risk_cluster + 1, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.legend()
        plt.show()
        
        return fig


# ============================================
# CLASS 4: TARGET VARIABLE INTEGRATOR
# ============================================
class TargetVariableIntegrator(BaseEstimator, TransformerMixin):
    """
    Instruction 4: Integrate the Target Variable
    
    Merge this new is_high_risk column back into your main processed dataset 
    for model training.
    """
    
    def __init__(self, customer_col='CustomerId'):
        """
        Parameters:
        -----------
        customer_col : str, default='CustomerId'
            Column to use for merging target variable
        """
        self.customer_col = customer_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, rfm_with_target):
        """
        Merge high-risk labels back to original/processed data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Original or processed dataset (transaction-level or customer-level)
        rfm_with_target : pd.DataFrame
            RFM data with is_high_risk labels (must have customer_col)
        """
        X = X.copy()
        
        # FIX: Ensure rfm_with_target has CustomerId as a column, not index
        if self.customer_col in rfm_with_target.index.names:
            # CustomerId is in the index, reset it
            rfm_with_target = rfm_with_target.reset_index()
        
        # Extract customer-risk mapping
        risk_mapping = rfm_with_target[[self.customer_col, 'is_high_risk']].drop_duplicates()
        
        # Instruction: "Merge this new is_high_risk column back into your main processed dataset"
        if self.customer_col in X.columns:
            X_with_target = pd.merge(
                X,
                risk_mapping,
                on=self.customer_col,
                how='left',
                validate='many_to_one' if X[self.customer_col].nunique() < len(X) else 'one_to_one'
            )
            
            # Fill any missing values (customers without RFM data)
            X_with_target['is_high_risk'] = X_with_target['is_high_risk'].fillna(0).astype(int)
            
            print(f" Target variable integrated: {X_with_target['is_high_risk'].sum():,} high-risk customers")
            print(f"📊 Dataset shape after integration: {X_with_target.shape}")
            print(f"🔍 Target distribution:")
            print(X_with_target['is_high_risk'].value_counts())
            print(f"📈 High-risk proportion: {(X_with_target['is_high_risk'].mean()*100):.2f}%")
            
            return X_with_target
        else:
            raise ValueError(f"Customer column '{self.customer_col}' not found in input data")
    
    def save_target_data(self, X_with_target, save_path=None):
        """Save dataset with target variable for model training"""
        import os
        from pathlib import Path
        
        if save_path is None:
            # Create default save path
            base_dir = Path(__file__).parent.parent
            save_path = base_dir / 'data' / 'processed' / 'task4_target_engineered.parquet'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save in multiple formats
        X_with_target.to_parquet(save_path, compression='snappy')
        csv_path = save_path.with_suffix('.csv')
        X_with_target.to_csv(csv_path, index=False)
        
        print(f"💾 Data saved successfully:")
        print(f"   📄 Parquet: {save_path}")
        print(f"   📊 CSV: {csv_path}")
        
        return save_path


# ============================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================
class TargetEngineeringPipeline:
    """Orchestrates all 4 steps of Task 4 target engineering"""
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.rfm_calculator = RFMMetricsCalculator()
        self.clusterer = RFMClustering(n_clusters=n_clusters, random_state=random_state)
        self.label_assigner = HighRiskLabelAssigner()
        self.integrator = TargetVariableIntegrator()
        self.final_data = None
        
    def run_pipeline(self, transaction_data, save_path=None):
        """Execute complete target engineering pipeline"""
        print("=" * 60)
        print("🎯 TASK 4: PROXY TARGET VARIABLE ENGINEERING")
        print("=" * 60)
        
        # Step 1: Calculate RFM Metrics
        print("\n📊 STEP 1: Calculating RFM Metrics...")
        rfm_data = self.rfm_calculator.transform(transaction_data)
        
        # Step 2: Cluster Customers
        print("\n🔍 STEP 2: Clustering Customers with K-Means...")
        self.clusterer.fit(rfm_data)
        rfm_with_clusters = self.clusterer.transform(rfm_data)
        
        # FIX: Ensure rfm_with_clusters has CustomerId as column for label assigner
        if 'CustomerId' in rfm_with_clusters.index.names:
            rfm_with_clusters = rfm_with_clusters.reset_index()
        
        # Step 3: Assign High-Risk Labels
        print("\n🏷️ STEP 3: Assigning High-Risk Labels...")
        self.label_assigner.fit(rfm_with_clusters, rfm_data.reset_index())
        rfm_with_target = self.label_assigner.transform(rfm_with_clusters)
        
        # Step 4: Integrate Target Variable
        print("\n🔄 STEP 4: Integrating Target Variable...")
        
        # FIX: Ensure rfm_with_target has CustomerId as column
        if 'CustomerId' not in rfm_with_target.columns:
            rfm_with_target = rfm_with_target.reset_index()
        
        self.final_data = self.integrator.transform(transaction_data, rfm_with_target)
        
        # Save results
        if save_path:
            self.integrator.save_target_data(self.final_data, save_path)
        
        print("\n" + "=" * 60)
        print("✅ TASK 4 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return self.final_data, rfm_with_target
    
    def get_pipeline_summary(self):
        """Get summary statistics of the pipeline execution"""
        if self.final_data is None:
            raise ValueError("Run pipeline first using run_pipeline()")
        
        summary = {
            'total_customers': self.final_data['CustomerId'].nunique(),
            'total_transactions': len(self.final_data),
            'high_risk_customers': self.final_data['is_high_risk'].sum(),
            'high_risk_proportion': self.final_data['is_high_risk'].mean(),
            'n_clusters': self.n_clusters,
            'silhouette_score': self.clusterer.silhouette_score,
            'high_risk_cluster': self.label_assigner.high_risk_cluster
        }
        
        return summary