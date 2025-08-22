"""
SignaMentis Data Cleaner Module

This module handles data preprocessing and cleaning for historical market data.
Includes outlier detection, missing value handling, and data validation.

Author: SignaMentis Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    UNUSABLE = "UNUSABLE"


class OutlierMethod(Enum):
    """Outlier detection methods."""
    IQR = "IQR"
    ZSCORE = "ZSCORE"
    ISOLATION_FOREST = "ISOLATION_FOREST"
    LOCAL_OUTLIER_FACTOR = "LOCAL_OUTLIER_FACTOR"
    DBSCAN = "DBSCAN"


@dataclass
class DataQualityReport:
    """Data quality report container."""
    timestamp: datetime
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    outliers: Dict[str, int]
    duplicates: int
    quality_score: float
    quality_level: DataQuality
    issues: List[str]
    recommendations: List[str]


class DataCleaner:
    """
    Data cleaner for preprocessing and cleaning historical market data.
    
    Features:
    - Missing value handling
    - Outlier detection and treatment
    - Data validation
    - Quality assessment
    - Statistical analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data cleaner.
        
        Args:
            config: Cleaner configuration
        """
        self.config = config or {}
        
        # Cleaning parameters
        self.outlier_method = OutlierMethod(self.config.get('outlier_method', 'IQR'))
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        self.min_data_quality = self.config.get('min_data_quality', 0.7)
        
        # Imputation parameters
        self.imputation_method = self.config.get('imputation_method', 'forward')
        self.max_gap_size = self.config.get('max_gap_size', 10)
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        
        # Validation parameters
        self.price_change_limit = self.config.get('price_change_limit', 0.1)  # 10%
        self.volume_change_limit = self.config.get('volume_change_limit', 0.5)  # 50%
        self.min_volume = self.config.get('min_volume', 0.0)
        self.max_price = self.config.get('max_price', 10000.0)
        self.min_price = self.config.get('min_price', 0.0)
        
        # Statistics tracking
        self.cleaning_stats = {}
        self.quality_reports = []
        
        logger.info("Data Cleaner initialized")
    
    def clean_data(self, data: pd.DataFrame, symbol: str = "XAUUSD") -> pd.DataFrame:
        """
        Clean and preprocess market data.
        
        Args:
            data: Raw market data DataFrame
            symbol: Trading symbol
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting data cleaning for {symbol}")
            
            # Create copy to avoid modifying original
            cleaned_data = data.copy()
            
            # Track original shape
            original_shape = cleaned_data.shape
            self.cleaning_stats['original_shape'] = original_shape
            
            # Step 1: Basic validation
            cleaned_data = self._validate_basic_structure(cleaned_data)
            
            # Step 2: Handle missing values
            cleaned_data = self._handle_missing_values(cleaned_data)
            
            # Step 3: Detect and handle outliers
            cleaned_data = self._handle_outliers(cleaned_data)
            
            # Step 4: Validate price and volume data
            cleaned_data = self._validate_price_volume(cleaned_data)
            
            # Step 5: Remove duplicates
            cleaned_data = self._remove_duplicates(cleaned_data)
            
            # Step 6: Sort by timestamp
            cleaned_data = self._sort_by_timestamp(cleaned_data)
            
            # Step 7: Reset index
            cleaned_data = cleaned_data.reset_index(drop=True)
            
            # Track final shape
            final_shape = cleaned_data.shape
            self.cleaning_stats['final_shape'] = final_shape
            self.cleaning_stats['rows_removed'] = original_shape[0] - final_shape[0]
            self.cleaning_stats['columns_removed'] = original_shape[1] - final_shape[1]
            
            logger.info(f"Data cleaning completed. Removed {self.cleaning_stats['rows_removed']} rows")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise
    
    def _validate_basic_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate basic data structure."""
        try:
            # Check required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Ensure numeric columns are numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove rows where all OHLC values are NaN
            ohlc_columns = ['open', 'high', 'low', 'close']
            data = data.dropna(subset=ohlc_columns, how='all')
            
            return data
            
        except Exception as e:
            logger.error(f"Error in basic structure validation: {e}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        try:
            # Count missing values
            missing_counts = data.isnull().sum()
            self.cleaning_stats['missing_values'] = missing_counts.to_dict()
            
            # Handle missing values in OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            
            for col in ohlc_columns:
                if data[col].isnull().sum() > 0:
                    # Forward fill for small gaps
                    data[col] = data[col].fillna(method='ffill', limit=self.max_gap_size)
                    
                    # Backward fill for remaining gaps
                    data[col] = data[col].fillna(method='bfill', limit=self.max_gap_size)
                    
                    # Interpolate for remaining gaps
                    if data[col].isnull().sum() > 0:
                        data[col] = data[col].interpolate(method=self.interpolation_method)
            
            # Handle missing volume values
            if data['volume'].isnull().sum() > 0:
                # Use median volume for missing values
                median_volume = data['volume'].median()
                data['volume'] = data['volume'].fillna(median_volume)
            
            # Remove rows with remaining NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in the data."""
        try:
            outlier_counts = {}
            
            # Handle outliers in OHLC columns
            ohlc_columns = ['open', 'high', 'low', 'close']
            
            for col in ohlc_columns:
                outliers = self._detect_outliers(data[col], col)
                outlier_counts[col] = len(outliers)
                
                if len(outliers) > 0:
                    # Replace outliers with interpolated values
                    data.loc[outliers.index, col] = np.nan
                    data[col] = data[col].interpolate(method=self.interpolation_method)
            
            # Handle volume outliers
            volume_outliers = self._detect_outliers(data['volume'], 'volume')
            outlier_counts['volume'] = len(volume_outliers)
            
            if len(volume_outliers) > 0:
                # Cap volume outliers at 3x median
                median_volume = data['volume'].median()
                max_volume = median_volume * 3
                data.loc[volume_outliers.index, 'volume'] = max_volume
            
            self.cleaning_stats['outliers_removed'] = outlier_counts
            
            return data
            
        except Exception as e:
            logger.error(f"Error handling outliers: {e}")
            raise
    
    def _detect_outliers(self, series: pd.Series, column_name: str) -> pd.Series:
        """Detect outliers using specified method."""
        try:
            if self.outlier_method == OutlierMethod.IQR:
                return self._detect_outliers_iqr(series)
            elif self.outlier_method == OutlierMethod.ZSCORE:
                return self._detect_outliers_zscore(series)
            elif self.outlier_method == OutlierMethod.ISOLATION_FOREST:
                return self._detect_outliers_isolation_forest(series)
            else:
                # Default to IQR method
                return self._detect_outliers_iqr(series)
                
        except Exception as e:
            logger.error(f"Error detecting outliers for {column_name}: {e}")
            return pd.Series(dtype=bool)
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            return outliers
            
        except Exception as e:
            logger.error(f"Error in IQR outlier detection: {e}")
            return pd.Series(dtype=bool)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        try:
            z_scores = np.abs(stats.zscore(series))
            outliers = z_scores > self.outlier_threshold
            return outliers
            
        except Exception as e:
            logger.error(f"Error in Z-score outlier detection: {e}")
            return pd.Series(dtype=bool)
    
    def _detect_outliers_isolation_forest(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Isolation Forest method."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X)
            
            # -1 indicates outliers
            outliers = predictions == -1
            return pd.Series(outliers, index=series.index)
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest outlier detection: {e}")
            return pd.Series(dtype=bool)
    
    def _validate_price_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate price and volume data."""
        try:
            initial_rows = len(data)
            
            # Validate OHLC relationships
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['open'] > data['high']) |
                (data['close'] > data['high']) |
                (data['open'] < data['low']) |
                (data['close'] < data['low'])
            )
            
            if invalid_ohlc.sum() > 0:
                logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
                data = data[~invalid_ohlc]
            
            # Validate price changes
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes > self.price_change_limit
            
            if extreme_changes.sum() > 0:
                logger.warning(f"Found {extreme_changes.sum()} rows with extreme price changes")
                # Mark for review but don't remove yet
                data['extreme_price_change'] = extreme_changes
            
            # Validate volume
            invalid_volume = (
                (data['volume'] < self.min_volume) |
                (data['volume'].isnull())
            )
            
            if invalid_volume.sum() > 0:
                logger.warning(f"Found {invalid_volume.sum()} rows with invalid volume")
                data = data[~invalid_volume]
            
            # Validate price ranges
            invalid_prices = (
                (data['close'] < self.min_price) |
                (data['close'] > self.max_price)
            )
            
            if invalid_prices.sum() > 0:
                logger.warning(f"Found {invalid_prices.sum()} rows with invalid prices")
                data = data[~invalid_prices]
            
            final_rows = len(data)
            rows_removed = initial_rows - final_rows
            
            if rows_removed > 0:
                logger.info(f"Removed {rows_removed} rows during price/volume validation")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in price/volume validation: {e}")
            raise
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        try:
            initial_rows = len(data)
            
            # Remove exact duplicates
            data = data.drop_duplicates()
            
            # Remove duplicates based on timestamp (keep first occurrence)
            data = data.drop_duplicates(subset=['timestamp'], keep='first')
            
            final_rows = len(data)
            duplicates_removed = initial_rows - final_rows
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate rows")
                self.cleaning_stats['duplicates_removed'] = duplicates_removed
            
            return data
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            raise
    
    def _sort_by_timestamp(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sort data by timestamp."""
        try:
            data = data.sort_values('timestamp')
            return data
            
        except Exception as e:
            logger.error(f"Error sorting by timestamp: {e}")
            raise
    
    def assess_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Assess the quality of the data.
        
        Args:
            data: DataFrame to assess
            
        Returns:
            DataQualityReport object
        """
        try:
            # Basic statistics
            total_rows = len(data)
            total_columns = len(data.columns)
            
            # Missing values analysis
            missing_values = data.isnull().sum().to_dict()
            missing_percentage = {col: (count / total_rows) * 100 for col, count in missing_values.items()}
            
            # Outlier analysis
            outliers = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col in data.columns:
                    outliers[col] = len(self._detect_outliers(data[col], col))
            
            # Duplicate analysis
            duplicates = data.duplicated().sum()
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                missing_percentage, outliers, duplicates, total_rows
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(quality_score)
            
            # Identify issues
            issues = self._identify_issues(missing_percentage, outliers, duplicates)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, quality_score)
            
            # Create report
            report = DataQualityReport(
                timestamp=datetime.now(),
                total_rows=total_rows,
                total_columns=total_columns,
                missing_values=missing_values,
                missing_percentage=missing_percentage,
                outliers=outliers,
                duplicates=duplicates,
                quality_score=quality_score,
                quality_level=quality_level,
                issues=issues,
                recommendations=recommendations
            )
            
            # Store report
            self.quality_reports.append(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            raise
    
    def _calculate_quality_score(self, missing_percentage: Dict, outliers: Dict, 
                                duplicates: int, total_rows: int) -> float:
        """Calculate overall data quality score."""
        try:
            # Missing value penalty (0-40 points)
            avg_missing = np.mean(list(missing_percentage.values()))
            missing_score = max(0, 40 - (avg_missing * 0.4))
            
            # Outlier penalty (0-30 points)
            total_outliers = sum(outliers.values())
            outlier_ratio = total_outliers / total_rows if total_rows > 0 else 0
            outlier_score = max(0, 30 - (outlier_ratio * 100))
            
            # Duplicate penalty (0-20 points)
            duplicate_ratio = duplicates / total_rows if total_rows > 0 else 0
            duplicate_score = max(0, 20 - (duplicate_ratio * 100))
            
            # Data completeness bonus (0-10 points)
            completeness_score = 10 if total_rows > 1000 else (total_rows / 100)
            
            total_score = missing_score + outlier_score + duplicate_score + completeness_score
            
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _determine_quality_level(self, quality_score: float) -> DataQuality:
        """Determine data quality level based on score."""
        if quality_score >= 90:
            return DataQuality.EXCELLENT
        elif quality_score >= 80:
            return DataQuality.GOOD
        elif quality_score >= 70:
            return DataQuality.FAIR
        elif quality_score >= 50:
            return DataQuality.POOR
        else:
            return DataQuality.UNUSABLE
    
    def _identify_issues(self, missing_percentage: Dict, outliers: Dict, 
                         duplicates: int) -> List[str]:
        """Identify data quality issues."""
        issues = []
        
        # Missing value issues
        for col, percentage in missing_percentage.items():
            if percentage > 5:
                issues.append(f"High missing values in {col}: {percentage:.1f}%")
        
        # Outlier issues
        for col, count in outliers.items():
            if count > 0:
                issues.append(f"Outliers detected in {col}: {count} values")
        
        # Duplicate issues
        if duplicates > 0:
            issues.append(f"Duplicate rows found: {duplicates}")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str], quality_score: float) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        if quality_score < 80:
            recommendations.append("Consider additional data cleaning steps")
        
        if any("missing values" in issue for issue in issues):
            recommendations.append("Implement more sophisticated imputation methods")
        
        if any("outliers" in issue for issue in issues):
            recommendations.append("Review outlier detection thresholds")
        
        if any("duplicate" in issue for issue in issues):
            recommendations.append("Investigate source of duplicate data")
        
        if quality_score < 70:
            recommendations.append("Data may not be suitable for trading without significant cleaning")
        
        return recommendations
    
    def generate_quality_report(self, data: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: DataFrame to analyze
            output_file: Optional output file path
            
        Returns:
            Report content as string
        """
        try:
            # Assess quality
            report = self.assess_data_quality(data)
            
            # Generate report content
            report_content = f"""
SignaMentis Data Quality Report
Generated: {report.timestamp}
{'='*50}

Data Overview:
- Total Rows: {report.total_rows:,}
- Total Columns: {report.total_columns}
- Quality Score: {report.quality_score:.1f}/100
- Quality Level: {report.quality_level.value}

Missing Values Analysis:
"""
            
            for col, percentage in report.missing_percentage.items():
                report_content += f"- {col}: {percentage:.2f}%\n"
            
            report_content += f"""
Outlier Analysis:
"""
            
            for col, count in report.outliers.items():
                report_content += f"- {col}: {count} outliers\n"
            
            report_content += f"""
Data Issues:
- Duplicates: {report.duplicates}
"""
            
            if report.issues:
                report_content += "\nIdentified Issues:\n"
                for issue in report.issues:
                    report_content += f"- {issue}\n"
            
            if report.recommendations:
                report_content += "\nRecommendations:\n"
                for rec in report.recommendations:
                    report_content += f"- {rec}\n"
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report_content)
                logger.info(f"Quality report saved to {output_file}")
            
            return report_content
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            raise
    
    def plot_data_quality(self, data: pd.DataFrame, output_file: Optional[str] = None):
        """
        Create visualizations for data quality analysis.
        
        Args:
            data: DataFrame to analyze
            output_file: Optional output file path
        """
        try:
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('SignaMentis Data Quality Analysis', fontsize=16)
            
            # Missing values heatmap
            missing_data = data.isnull()
            sns.heatmap(missing_data, cbar=True, ax=axes[0, 0])
            axes[0, 0].set_title('Missing Values Heatmap')
            
            # Missing values bar plot
            missing_counts = data.isnull().sum()
            missing_counts.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values by Column')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Price distribution
            data['close'].hist(bins=50, ax=axes[1, 0])
            axes[1, 0].set_title('Close Price Distribution')
            axes[1, 0].set_xlabel('Price')
            axes[1, 0].set_ylabel('Frequency')
            
            # Volume distribution
            data['volume'].hist(bins=50, ax=axes[1, 1])
            axes[1, 1].set_title('Volume Distribution')
            axes[1, 1].set_xlabel('Volume')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            # Save plot if specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Quality plot saved to {output_file}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating quality plots: {e}")
            raise
    
    def get_cleaning_statistics(self) -> Dict[str, Any]:
        """Get cleaning statistics."""
        return self.cleaning_stats.copy()
    
    def get_quality_reports(self) -> List[DataQualityReport]:
        """Get all quality reports."""
        return self.quality_reports.copy()


# Convenience function for creating cleaner
def create_data_cleaner(config: Optional[Dict] = None) -> DataCleaner:
    """
    Create a data cleaner instance with configuration.
    
    Args:
        config: Cleaner configuration dictionary
        
    Returns:
        DataCleaner instance
    """
    return DataCleaner(config)


if __name__ == "__main__":
    # Example usage
    config = {
        'outlier_method': 'IQR',
        'outlier_threshold': 3.0,
        'iqr_multiplier': 1.5,
        'min_data_quality': 0.7,
        'imputation_method': 'forward',
        'max_gap_size': 10,
        'interpolation_method': 'linear'
    }
    
    # Create cleaner
    cleaner = create_data_cleaner(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='5T')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(2000, 50, 1000),
        'high': np.random.normal(2010, 50, 1000),
        'low': np.random.normal(1990, 50, 1000),
        'close': np.random.normal(2000, 50, 1000),
        'volume': np.random.normal(100, 20, 1000)
    })
    
    # Add some data quality issues
    sample_data.loc[100:110, 'close'] = np.nan  # Missing values
    sample_data.loc[200, 'close'] = 5000  # Outlier
    sample_data.loc[300:310] = sample_data.iloc[100:110]  # Duplicates
    
    print("Sample data created with quality issues")
    print(f"Original shape: {sample_data.shape}")
    
    # Clean data
    cleaned_data = cleaner.clean_data(sample_data, "XAUUSD")
    print(f"Cleaned shape: {cleaned_data.shape}")
    
    # Assess quality
    quality_report = cleaner.assess_data_quality(cleaned_data)
    print(f"Data quality: {quality_report.quality_level.value} ({quality_report.quality_score:.1f}/100)")
    
    # Generate report
    report_content = cleaner.generate_quality_report(cleaned_data)
    print("\nQuality Report:")
    print(report_content)
    
    # Get statistics
    stats = cleaner.get_cleaning_statistics()
    print(f"\nCleaning Statistics: {stats}")
    
    print("Data Cleaner test completed!")
