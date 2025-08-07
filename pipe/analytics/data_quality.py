"""
Data Quality Assessment Module

Comprehensive data quality analysis and validation for professional data analytics.
Essential for ensuring data reliability and accuracy in business intelligence.

Author: Osman Abdullahi
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class DataQualityAssessment:
    """
    Professional data quality assessment toolkit for data analysts.
    
    Provides comprehensive data quality analysis including:
    - Completeness analysis
    - Consistency validation
    - Accuracy assessment
    - Uniqueness checks
    - Validity testing
    - Data profiling
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Data Quality Assessment.
        
        Args:
            data (pd.DataFrame): Dataset to assess
        """
        self.data = data.copy()
        self.quality_report = {}
        self.issues = []
        
    def assess_completeness(self) -> Dict:
        """
        Assess data completeness - missing values analysis.
        
        Returns:
            Dict: Completeness assessment results
        """
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_cells = self.data.isnull().sum().sum()
        completeness_rate = ((total_cells - missing_cells) / total_cells) * 100
        
        # Column-wise completeness
        column_completeness = {}
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            completeness = ((len(self.data) - missing_count) / len(self.data)) * 100
            column_completeness[col] = {
                'completeness_rate': completeness,
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(self.data)) * 100
            }
            
            # Flag columns with high missing rates
            if completeness < 80:
                self.issues.append(f"Column '{col}' has low completeness: {completeness:.1f}%")
        
        completeness_results = {
            'overall_completeness_rate': completeness_rate,
            'total_missing_cells': missing_cells,
            'column_completeness': column_completeness,
            'assessment_date': datetime.now().isoformat()
        }
        
        self.quality_report['completeness'] = completeness_results
        return completeness_results
    
    def assess_consistency(self) -> Dict:
        """
        Assess data consistency - format and pattern validation.
        
        Returns:
            Dict: Consistency assessment results
        """
        consistency_results = {}
        
        for col in self.data.columns:
            col_data = self.data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            col_analysis = {
                'data_type': str(col_data.dtype),
                'unique_values': col_data.nunique(),
                'sample_values': col_data.head(5).tolist()
            }
            
            # Check for mixed data types in object columns
            if col_data.dtype == 'object':
                type_counts = {}
                for value in col_data.head(100):  # Sample for performance
                    value_type = type(value).__name__
                    type_counts[value_type] = type_counts.get(value_type, 0) + 1
                
                col_analysis['value_types'] = type_counts
                
                # Flag mixed types
                if len(type_counts) > 1:
                    self.issues.append(f"Column '{col}' has mixed data types: {list(type_counts.keys())}")
            
            # Check for outliers in numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_percentage = (len(outliers) / len(col_data)) * 100
                
                col_analysis['outliers'] = {
                    'count': len(outliers),
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                # Flag high outlier rates
                if outlier_percentage > 5:
                    self.issues.append(f"Column '{col}' has high outlier rate: {outlier_percentage:.1f}%")
            
            consistency_results[col] = col_analysis
        
        self.quality_report['consistency'] = consistency_results
        return consistency_results
    
    def assess_uniqueness(self) -> Dict:
        """
        Assess data uniqueness - duplicate detection.
        
        Returns:
            Dict: Uniqueness assessment results
        """
        # Overall duplicate analysis
        total_rows = len(self.data)
        duplicate_rows = self.data.duplicated().sum()
        uniqueness_rate = ((total_rows - duplicate_rows) / total_rows) * 100
        
        # Column-wise uniqueness
        column_uniqueness = {}
        for col in self.data.columns:
            col_data = self.data[col].dropna()
            unique_count = col_data.nunique()
            total_count = len(col_data)
            uniqueness = (unique_count / total_count) * 100 if total_count > 0 else 0
            
            column_uniqueness[col] = {
                'unique_count': unique_count,
                'total_count': total_count,
                'uniqueness_rate': uniqueness,
                'duplicate_count': total_count - unique_count
            }
            
            # Flag potential ID columns with low uniqueness
            if 'id' in col.lower() and uniqueness < 95:
                self.issues.append(f"ID column '{col}' has low uniqueness: {uniqueness:.1f}%")
        
        uniqueness_results = {
            'overall_uniqueness_rate': uniqueness_rate,
            'total_duplicate_rows': duplicate_rows,
            'column_uniqueness': column_uniqueness
        }
        
        self.quality_report['uniqueness'] = uniqueness_results
        return uniqueness_results
    
    def assess_validity(self) -> Dict:
        """
        Assess data validity - business rule validation.
        
        Returns:
            Dict: Validity assessment results
        """
        validity_results = {}
        
        for col in self.data.columns:
            col_data = self.data[col].dropna()
            col_validity = {'valid_count': len(col_data), 'invalid_count': 0, 'issues': []}
            
            # Date validation
            if 'date' in col.lower() or col_data.dtype == 'datetime64[ns]':
                try:
                    if col_data.dtype != 'datetime64[ns]':
                        pd.to_datetime(col_data, errors='raise')
                    
                    # Check for future dates if it's a transaction/historical date
                    if any(keyword in col.lower() for keyword in ['transaction', 'created', 'order']):
                        future_dates = col_data[col_data > datetime.now()]
                        if len(future_dates) > 0:
                            col_validity['issues'].append(f"Found {len(future_dates)} future dates")
                            col_validity['invalid_count'] += len(future_dates)
                            
                except Exception:
                    col_validity['issues'].append("Invalid date format detected")
            
            # Email validation (basic)
            if 'email' in col.lower():
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = col_data[~col_data.str.match(email_pattern, na=False)]
                if len(invalid_emails) > 0:
                    col_validity['issues'].append(f"Found {len(invalid_emails)} invalid email formats")
                    col_validity['invalid_count'] += len(invalid_emails)
            
            # Numeric range validation
            if pd.api.types.is_numeric_dtype(col_data):
                # Age validation
                if 'age' in col.lower():
                    invalid_ages = col_data[(col_data < 0) | (col_data > 150)]
                    if len(invalid_ages) > 0:
                        col_validity['issues'].append(f"Found {len(invalid_ages)} invalid ages")
                        col_validity['invalid_count'] += len(invalid_ages)
                
                # Revenue/Price validation
                if any(keyword in col.lower() for keyword in ['revenue', 'price', 'amount', 'cost']):
                    negative_values = col_data[col_data < 0]
                    if len(negative_values) > 0:
                        col_validity['issues'].append(f"Found {len(negative_values)} negative financial values")
                        col_validity['invalid_count'] += len(negative_values)
            
            validity_results[col] = col_validity
            
            # Add to global issues
            for issue in col_validity['issues']:
                self.issues.append(f"Column '{col}': {issue}")
        
        self.quality_report['validity'] = validity_results
        return validity_results
    
    def generate_data_profile(self) -> Dict:
        """
        Generate comprehensive data profile.
        
        Returns:
            Dict: Complete data profile
        """
        profile = {
            'dataset_overview': {
                'total_rows': len(self.data),
                'total_columns': len(self.data.columns),
                'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
                'column_types': self.data.dtypes.value_counts().to_dict()
            },
            'column_profiles': {}
        }
        
        for col in self.data.columns:
            col_data = self.data[col]
            
            col_profile = {
                'data_type': str(col_data.dtype),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'memory_usage': col_data.memory_usage(deep=True)
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                col_profile.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75)
                })
            
            # Add top values for categorical columns
            if col_data.dtype == 'object' or col_data.nunique() < 20:
                col_profile['top_values'] = col_data.value_counts().head(10).to_dict()
            
            profile['column_profiles'][col] = col_profile
        
        self.quality_report['data_profile'] = profile
        return profile
    
    def run_comprehensive_assessment(self) -> Dict:
        """
        Run complete data quality assessment.
        
        Returns:
            Dict: Comprehensive quality assessment report
        """
        print("Running comprehensive data quality assessment...")
        
        # Run all assessments
        self.assess_completeness()
        self.assess_consistency()
        self.assess_uniqueness()
        self.assess_validity()
        self.generate_data_profile()
        
        # Calculate overall quality score
        completeness_score = self.quality_report['completeness']['overall_completeness_rate']
        uniqueness_score = self.quality_report['uniqueness']['overall_uniqueness_rate']
        
        # Simple scoring (can be enhanced)
        overall_score = (completeness_score + uniqueness_score) / 2
        
        # Determine quality level
        if overall_score >= 90:
            quality_level = "Excellent"
        elif overall_score >= 75:
            quality_level = "Good"
        elif overall_score >= 60:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        # Summary report
        summary = {
            'overall_quality_score': overall_score,
            'quality_level': quality_level,
            'total_issues': len(self.issues),
            'critical_issues': [issue for issue in self.issues if any(word in issue.lower() 
                               for word in ['low completeness', 'invalid', 'mixed types'])],
            'assessment_timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations()
        }
        
        self.quality_report['summary'] = summary
        
        print(f"Assessment complete. Overall quality score: {overall_score:.1f}% ({quality_level})")
        print(f"Found {len(self.issues)} data quality issues.")
        
        return self.quality_report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if 'completeness' in self.quality_report:
            completeness = self.quality_report['completeness']['overall_completeness_rate']
            if completeness < 90:
                recommendations.append("Improve data collection processes to reduce missing values")
                recommendations.append("Implement data validation at source systems")
        
        # Consistency recommendations
        if len([issue for issue in self.issues if 'mixed types' in issue]) > 0:
            recommendations.append("Standardize data types and formats across all data sources")
            recommendations.append("Implement data transformation rules for consistent formatting")
        
        # Validity recommendations
        if len([issue for issue in self.issues if 'invalid' in issue.lower()]) > 0:
            recommendations.append("Implement business rule validation in data pipelines")
            recommendations.append("Add data quality checks before analysis")
        
        # General recommendations
        recommendations.append("Schedule regular data quality assessments")
        recommendations.append("Create data quality monitoring dashboards")
        
        return recommendations
    
    def export_report(self, filename: str = None) -> str:
        """
        Export quality assessment report to file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            str: Report content
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_quality_report_{timestamp}.md"
        
        report_content = "# Data Quality Assessment Report\n\n"
        report_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_content += f"**Dataset**: {self.data.shape[0]} rows, {self.data.shape[1]} columns\n\n"
        
        if 'summary' in self.quality_report:
            summary = self.quality_report['summary']
            report_content += "## Executive Summary\n\n"
            report_content += f"- **Overall Quality Score**: {summary['overall_quality_score']:.1f}%\n"
            report_content += f"- **Quality Level**: {summary['quality_level']}\n"
            report_content += f"- **Total Issues Found**: {summary['total_issues']}\n\n"
        
        if self.issues:
            report_content += "## Key Issues Identified\n\n"
            for issue in self.issues[:10]:  # Top 10 issues
                report_content += f"- {issue}\n"
            report_content += "\n"
        
        if 'summary' in self.quality_report and 'recommendations' in self.quality_report['summary']:
            report_content += "## Recommendations\n\n"
            for rec in self.quality_report['summary']['recommendations']:
                report_content += f"- {rec}\n"
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(report_content)
        
        return report_content
