"""
EDA Utility Functions for Credit Risk Analysis
Contains reusable functions for exploratory data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """Class for performing exploratory data analysis"""
    
    def __init__(self, df):
        """
        Initialize with dataframe
        
        Parameters:
        -----------
        df : pandas DataFrame
            The dataset to analyze
        """
        self.df = df.copy()
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self._detect_column_types()
        
    def _detect_column_types(self):
        """Detect column types automatically"""
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.datetime_cols.append(col)
            else:
                self.categorical_cols.append(col)
    
    def get_data_overview(self):
        """
        Get comprehensive overview of the dataset
        
        Returns:
        --------
        dict: Dictionary containing overview statistics
        """
        overview = {
            'shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_percentage': (self.df.duplicated().sum() / len(self.df)) * 100,
            'column_types': {
                'numeric': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'datetime': len(self.datetime_cols)
            }
        }
        return overview
    
    def get_missing_values_report(self):
        """
        Analyze missing values in the dataset
        
        Returns:
        --------
        pandas DataFrame: Missing values statistics
        """
        missing_stats = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            # For numeric columns, get additional stats
            if col in self.numeric_cols:
                col_mean = self.df[col].mean() if missing_count < len(self.df) else np.nan
                col_median = self.df[col].median() if missing_count < len(self.df) else np.nan
            else:
                col_mean = col_median = np.nan
            
            missing_stats.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'data_type': str(self.df[col].dtype),
                'unique_values': self.df[col].nunique(),
                'mean': col_mean,
                'median': col_median
            })
        
        return pd.DataFrame(missing_stats)
    
    def plot_missing_values(self, figsize=(12, 6)):
        """
        Visualize missing values
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        missing_df = self.get_missing_values_report()
        missing_df = missing_df[missing_df['missing_count'] > 0]
        
        if missing_df.empty:
            print("No missing values found in the dataset.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        bars = ax1.barh(missing_df['column'], missing_df['missing_percentage'])
        ax1.set_xlabel('Missing Percentage (%)')
        ax1.set_title('Missing Values by Column')
        ax1.invert_yaxis()
        
        # Add percentage labels
        for bar, pct in zip(bars, missing_df['missing_percentage']):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center')
        
        # Heatmap
        missing_matrix = self.df.isnull()
        sns.heatmap(missing_matrix, cbar=False, cmap='viridis', ax=ax2)
        ax2.set_title('Missing Values Pattern')
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Rows')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_numeric_features(self, plot=True):
        """
        Analyze numerical features
        
        Parameters:
        -----------
        plot : bool
            Whether to create visualization plots
            
        Returns:
        --------
        pandas DataFrame: Statistics for numerical features
        """
        if not self.numeric_cols:
            print("No numeric columns found.")
            return pd.DataFrame()
        
        stats = []
        for col in self.numeric_cols:
            col_stats = {
                'column': col,
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75),
                'iqr': self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'zeros': (self.df[col] == 0).sum(),
                'zeros_percentage': ((self.df[col] == 0).sum() / len(self.df)) * 100
            }
            stats.append(col_stats)
        
        stats_df = pd.DataFrame(stats)
        
        if plot and not self.numeric_cols.empty:
            self._plot_numeric_distributions()
            
        return stats_df
    
    def _plot_numeric_distributions(self, cols_per_row=3):
        """Plot distributions for numeric features"""
        n_cols = len(self.numeric_cols)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.numeric_cols):
            ax = axes[idx]
            
            # Histogram with KDE
            sns.histplot(data=self.df, x=col, kde=True, ax=ax, bins=50)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-.', alpha=0.7, label=f'Median: {median_val:.2f}')
            ax.legend()
        
        # Hide empty subplots
        for idx in range(len(self.numeric_cols), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def analyze_categorical_features(self, top_n=10, plot=True):
        """
        Analyze categorical features
        
        Parameters:
        -----------
        top_n : int
            Number of top categories to show
        plot : bool
            Whether to create visualization plots
            
        Returns:
        --------
        dict: Statistics for categorical features
        """
        if not self.categorical_cols:
            print("No categorical columns found.")
            return {}
        
        stats = {}
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            stats[col] = {
                'unique_values': self.df[col].nunique(),
                'missing_values': self.df[col].isnull().sum(),
                'top_categories': value_counts.head(top_n).to_dict(),
                'value_counts': value_counts
            }
        
        if plot:
            self._plot_categorical_distributions(top_n)
            
        return stats
    
    def _plot_categorical_distributions(self, top_n=10):
        """Plot distributions for categorical features"""
        n_cols = len(self.categorical_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.categorical_cols):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            value_counts = self.df[col].value_counts().head(top_n)
            
            if len(value_counts) > 20:  # For high cardinality, use bar plot
                bars = ax.bar(range(len(value_counts)), value_counts.values)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            else:
                # Create pie chart for low cardinality
                wedges, texts, autotexts = ax.pie(
                    value_counts.values, 
                    labels=value_counts.index,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as circle
            
            ax.set_title(f'Distribution of {col}\n(Unique: {self.df[col].nunique()})')
        
        # Hide empty subplots
        for idx in range(len(self.categorical_cols), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers in numerical features
        
        Parameters:
        -----------
        method : str
            'iqr' for Interquartile Range or 'zscore' for Z-score method
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        dict: Outlier statistics for each numeric column
        """
        outliers_dict = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(data))
                outliers = data[z_scores > threshold]
            
            outliers_dict[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(data)) * 100,
                'min_outlier': outliers.min() if not outliers.empty else np.nan,
                'max_outlier': outliers.max() if not outliers.empty else np.nan,
                'outliers': outliers.tolist() if len(outliers) <= 10 else outliers.head(10).tolist()
            }
        
        return outliers_dict
    
    def plot_outliers(self, method='iqr', threshold=1.5):
        """
        Visualize outliers using box plots
        
        Parameters:
        -----------
        method : str
            Detection method
        threshold : float
            Threshold for outlier detection
        """
        n_cols = len(self.numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        outliers_dict = self.detect_outliers(method, threshold)
        
        for idx, col in enumerate(self.numeric_cols):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Box plot
            bp = ax.boxplot(self.df[col].dropna(), patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['medians'][0].set_color('red')
            
            # Add scatter for outliers
            outliers = self.df[col][self.df[col].isin(
                outliers_dict[col]['outliers']
            )].dropna()
            
            if not outliers.empty:
                ax.scatter(
                    np.ones(len(outliers)), 
                    outliers.values, 
                    color='red', 
                    alpha=0.6, 
                    label='Outliers'
                )
            
            ax.set_title(f'{col}\nOutliers: {outliers_dict[col]["outlier_count"]} ({outliers_dict[col]["outlier_percentage"]:.1f}%)')
            ax.set_ylabel(col)
            ax.set_xticks([])
            
            if idx == 0:
                ax.legend()
        
        # Hide empty subplots
        for idx in range(len(self.numeric_cols), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def analyze_correlations(self, method='pearson', plot=True):
        """
        Analyze correlations between numerical features
        
        Parameters:
        -----------
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')
        plot : bool
            Whether to plot correlation matrix
            
        Returns:
        --------
        pandas DataFrame: Correlation matrix
        """
        if len(self.numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation analysis.")
            return pd.DataFrame()
        
        corr_matrix = self.df[self.numeric_cols].corr(method=method)
        
        if plot:
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix, 
                mask=mask, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            plt.title(f'Correlation Matrix ({method.capitalize()})')
            plt.tight_layout()
            plt.show()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print("\nHighly Correlated Feature Pairs (|corr| > 0.7):")
            for pair in high_corr_pairs:
                print(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}")
        
        return corr_matrix
    
    def analyze_temporal_features(self):
        """
        Analyze datetime features for temporal patterns
        """
        if not self.datetime_cols:
            # Try to identify datetime columns from TransactionStartTime
            if 'TransactionStartTime' in self.df.columns:
                self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
                self.datetime_cols.append('TransactionStartTime')
            else:
                print("No datetime columns found.")
                return {}
        
        temporal_stats = {}
        
        for col in self.datetime_cols:
            temporal_stats[col] = {
                'min_date': self.df[col].min(),
                'max_date': self.df[col].max(),
                'date_range_days': (self.df[col].max() - self.df[col].min()).days,
                'unique_dates': self.df[col].dt.date.nunique()
            }
        
        # Plot temporal patterns
        if 'TransactionStartTime' in self.df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Daily transaction count
            daily_counts = self.df.set_index('TransactionStartTime').resample('D').size()
            axes[0, 0].plot(daily_counts.index, daily_counts.values)
            axes[0, 0].set_title('Daily Transaction Volume')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Transaction Count')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Hourly distribution
            self.df['hour'] = self.df['TransactionStartTime'].dt.hour
            hourly_counts = self.df['hour'].value_counts().sort_index()
            axes[0, 1].bar(hourly_counts.index, hourly_counts.values)
            axes[0, 1].set_title('Hourly Transaction Distribution')
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Transaction Count')
            axes[0, 1].set_xticks(range(0, 24, 2))
            
            # Day of week distribution
            self.df['day_of_week'] = self.df['TransactionStartTime'].dt.day_name()
            dow_counts = self.df['day_of_week'].value_counts()
            axes[1, 0].bar(dow_counts.index, dow_counts.values)
            axes[1, 0].set_title('Transactions by Day of Week')
            axes[1, 0].set_xlabel('Day')
            axes[1, 0].set_ylabel('Transaction Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Monthly distribution
            self.df['month'] = self.df['TransactionStartTime'].dt.month_name()
            monthly_counts = self.df['month'].value_counts()
            axes[1, 1].bar(monthly_counts.index, monthly_counts.values)
            axes[1, 1].set_title('Transactions by Month')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Transaction Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        return temporal_stats
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        
        Returns:
        --------
        dict: Complete summary of the dataset
        """
        report = {
            'data_overview': self.get_data_overview(),
            'missing_values': self.get_missing_values_report(),
            'numeric_stats': self.analyze_numeric_features(plot=False),
            'categorical_stats': self.analyze_categorical_features(plot=False),
            'temporal_stats': self.analyze_temporal_features() if self.datetime_cols else {},
            'correlations': self.analyze_correlations(plot=False) if len(self.numeric_cols) >= 2 else {},
            'outliers': self.detect_outliers()
        }
        
        return report