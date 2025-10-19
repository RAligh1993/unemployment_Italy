"""
🔍 Universal Data File Analyzer v1.0
=====================================
Automatic detection and analysis for any data file:
- File types: CSV, Excel (XLS, XLSX), TXT, TSV
- Frequencies: Daily, Weekly, Monthly, Quarterly, Annual
- Features: Automatic detection of date/numeric columns
- Quality: Comprehensive data quality assessment

Author: AI Assistant
Date: October 2025
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


class DataFileAnalyzer:
    """
    Comprehensive analyzer for any data file
    
    Capabilities:
    - Auto-detect file format
    - Auto-detect date columns
    - Auto-detect frequency
    - Extract all numeric features
    - Quality assessment
    - Format recommendations
    """
    
    # Date column patterns
    DATE_PATTERNS = [
        'date', 'Date', 'DATE', 'ds', 'time', 'Time', 'TIME',
        'period', 'Period', 'PERIOD', 'timestamp', 'Timestamp',
        'week', 'Week', 'WEEK', 'month', 'Month', 'MONTH',
        'quarter', 'Quarter', 'QUARTER', 'year', 'Year', 'YEAR',
        'day', 'Day', 'DAY', 'fecha', 'data', 'datum'
    ]
    
    # Quarterly formats
    QUARTERLY_PATTERNS = [
        r'(\d{4})[-\s]?Q([1-4])',           # 2020-Q1, 2020Q1
        r'(\d{4})[-\s]?q([1-4])',           # 2020-q1
        r'Q([1-4])[-\s]?(\d{4})',           # Q1-2020
        r'(\d{4})[\.]\s?([1-4])',           # 2020.1
        r'(\d{4})[-/]?\s?0?([1-4])',        # 2020-01, 2020/1
    ]
    
    # Frequency definitions (median days ± tolerance)
    FREQUENCY_MAP = [
        (1, 'daily', 2),
        (7, 'weekly', 3),
        (14, 'biweekly', 4),
        (30, 'monthly', 6),
        (91, 'quarterly', 15),
        (182, 'semiannual', 20),
        (365, 'annual', 40),
    ]
    
    def __init__(self, file_path: Union[str, BytesIO], encoding: str = 'utf-8'):
        """
        Initialize analyzer
        
        Args:
            file_path: Path to file or BytesIO object
            encoding: Default encoding for text files
        """
        self.file_path = file_path
        self.encoding = encoding
        self.file_name = Path(file_path).name if isinstance(file_path, str) else 'uploaded_file'
        self.file_ext = Path(self.file_name).suffix.lower()
        
        # Analysis results
        self.analysis = {}
        self.dataframes = {}  # For Excel with multiple sheets
        self.errors = []
        
    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis pipeline
        
        Returns:
            Comprehensive analysis dictionary
        """
        print(f"\n{'='*70}")
        print(f"📊 ANALYZING FILE: {self.file_name}")
        print(f"{'='*70}\n")
        
        # Step 1: Detect and read file
        print("🔍 Step 1: Reading file...")
        self._read_file()
        
        if self.errors:
            return self._create_error_report()
        
        # Step 2: Analyze each sheet/dataframe
        print("\n🔍 Step 2: Analyzing structure...")
        self._analyze_structure()
        
        # Step 3: Detect date columns
        print("\n🔍 Step 3: Detecting date columns...")
        self._detect_date_columns()
        
        # Step 4: Detect frequency
        print("\n🔍 Step 4: Detecting frequency...")
        self._detect_frequency()
        
        # Step 5: Extract features
        print("\n🔍 Step 5: Extracting features...")
        self._extract_features()
        
        # Step 6: Quality assessment
        print("\n🔍 Step 6: Assessing quality...")
        self._assess_quality()
        
        # Step 7: Generate recommendations
        print("\n🔍 Step 7: Generating recommendations...")
        self._generate_recommendations()
        
        print("\n✅ Analysis complete!\n")
        
        return self.analysis
    
    def _read_file(self):
        """Read file based on extension"""
        try:
            if self.file_ext in ['.csv', '.txt', '.tsv']:
                self._read_text_file()
            elif self.file_ext in ['.xlsx', '.xls', '.xlsm']:
                self._read_excel_file()
            else:
                self.errors.append(f"Unsupported file format: {self.file_ext}")
        
        except Exception as e:
            self.errors.append(f"Error reading file: {str(e)}")
    
    def _read_text_file(self):
        """Read CSV/TXT/TSV with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        delimiters = [',', ';', '\t', '|']
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    if isinstance(self.file_path, BytesIO):
                        self.file_path.seek(0)
                        df = pd.read_csv(self.file_path, encoding=encoding, sep=delimiter)
                    else:
                        df = pd.read_csv(self.file_path, encoding=encoding, sep=delimiter)
                    
                    # Check if valid (more than 1 column)
                    if len(df.columns) > 1:
                        self.dataframes['main'] = df
                        self.analysis['encoding'] = encoding
                        self.analysis['delimiter'] = delimiter
                        print(f"   ✅ Read with encoding={encoding}, delimiter='{delimiter}'")
                        return
                
                except Exception:
                    continue
        
        self.errors.append("Could not read file with any encoding/delimiter combination")
    
    def _read_excel_file(self):
        """Read Excel file with all sheets"""
        try:
            if isinstance(self.file_path, BytesIO):
                self.file_path.seek(0)
                excel_file = pd.ExcelFile(self.file_path)
            else:
                excel_file = pd.ExcelFile(self.file_path)
            
            sheet_names = excel_file.sheet_names
            
            print(f"   📄 Found {len(sheet_names)} sheet(s): {sheet_names}")
            
            for sheet_name in sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                if not df.empty:
                    self.dataframes[sheet_name] = df
                    print(f"   ✅ Loaded sheet: '{sheet_name}' ({len(df)} rows, {len(df.columns)} cols)")
            
            if not self.dataframes:
                self.errors.append("All sheets are empty")
        
        except Exception as e:
            self.errors.append(f"Error reading Excel: {str(e)}")
    
    def _analyze_structure(self):
        """Analyze basic structure of each dataframe"""
        self.analysis['sheets'] = {}
        
        for sheet_name, df in self.dataframes.items():
            structure = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'null_counts': df.isnull().sum().to_dict(),
            }
            
            self.analysis['sheets'][sheet_name] = structure
            
            print(f"   📊 {sheet_name}: {structure['rows']} rows × {structure['columns']} cols")
    
    def _detect_date_columns(self):
        """Detect date columns in each sheet"""
        for sheet_name, df in self.dataframes.items():
            date_candidates = []
            
            # Method 1: Check column names
            for col in df.columns:
                col_lower = str(col).lower()
                
                # Check against patterns
                for pattern in self.DATE_PATTERNS:
                    if pattern.lower() in col_lower:
                        date_candidates.append({
                            'column': col,
                            'method': 'name_pattern',
                            'confidence': 0.9
                        })
                        break
            
            # Method 2: Check data types
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    if not any(c['column'] == col for c in date_candidates):
                        date_candidates.append({
                            'column': col,
                            'method': 'datetime_dtype',
                            'confidence': 1.0
                        })
            
            # Method 3: Try parsing first few values
            for col in df.columns:
                if col in [c['column'] for c in date_candidates]:
                    continue
                
                try:
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        parsed = pd.to_datetime(sample, errors='coerce')
                        parse_rate = parsed.notna().sum() / len(sample)
                        
                        if parse_rate > 0.7:
                            date_candidates.append({
                                'column': col,
                                'method': 'parse_test',
                                'confidence': parse_rate
                            })
                except Exception:
                    pass
            
            # Method 4: Check for quarterly patterns
            for col in df.columns:
                if col in [c['column'] for c in date_candidates]:
                    continue
                
                sample = df[col].astype(str).head(10)
                
                for pattern in self.QUARTERLY_PATTERNS:
                    if sample.str.contains(pattern, regex=True).any():
                        date_candidates.append({
                            'column': col,
                            'method': 'quarterly_pattern',
                            'confidence': 0.95
                        })
                        break
            
            # Sort by confidence
            date_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.analysis['sheets'][sheet_name]['date_candidates'] = date_candidates
            
            if date_candidates:
                primary = date_candidates[0]
                print(f"   📅 {sheet_name}: Date column = '{primary['column']}' (confidence: {primary['confidence']:.0%})")
            else:
                print(f"   ⚠️  {sheet_name}: No date column detected")
    
    def _detect_frequency(self):
        """Detect time series frequency for each sheet"""
        for sheet_name, df in self.dataframes.items():
            sheet_analysis = self.analysis['sheets'][sheet_name]
            date_candidates = sheet_analysis.get('date_candidates', [])
            
            if not date_candidates:
                sheet_analysis['frequency'] = {
                    'detected': 'unknown',
                    'confidence': 0.0,
                    'reason': 'No date column found'
                }
                continue
            
            # Use best date column
            date_col = date_candidates[0]['column']
            date_series = df[date_col]
            
            # Check for quarterly format
            is_quarterly = self._is_quarterly_format(date_series)
            
            if is_quarterly:
                freq_result = self._analyze_quarterly_frequency(date_series)
            else:
                freq_result = self._analyze_regular_frequency(date_series)
            
            sheet_analysis['frequency'] = freq_result
            
            print(f"   🔍 {sheet_name}: {freq_result['detected']} (confidence: {freq_result['confidence']:.0%})")
    
    def _is_quarterly_format(self, series: pd.Series) -> bool:
        """Check if series contains quarterly format strings"""
        sample = series.astype(str).head(20)
        
        for pattern in self.QUARTERLY_PATTERNS:
            if sample.str.contains(pattern, regex=True).any():
                return True
        
        return False
    
    def _analyze_quarterly_frequency(self, series: pd.Series) -> Dict:
        """Analyze quarterly data frequency"""
        # Parse quarterly strings
        dates = []
        
        for pattern in self.QUARTERLY_PATTERNS:
            try:
                extracted = series.astype(str).str.extract(pattern)
                
                if not extracted.empty and not extracted[0].isna().all():
                    years = extracted[0].astype(float)
                    quarters = extracted[1].astype(float)
                    
                    # Convert to quarter-end dates
                    parsed_dates = pd.to_datetime(
                        years.astype(int).astype(str) + '-' + 
                        (quarters.astype(int) * 3).astype(str) + '-01'
                    ) + pd.offsets.QuarterEnd(0)
                    
                    dates = parsed_dates.dropna().sort_values()
                    break
            
            except Exception:
                continue
        
        if len(dates) < 2:
            return {
                'detected': 'quarterly',
                'confidence': 0.5,
                'reason': 'Quarterly format detected but insufficient data',
                'observations': len(dates)
            }
        
        # Calculate gaps
        gaps_months = dates.diff().dt.days / 30.44
        median_gap = gaps_months.median()
        std_gap = gaps_months.std()
        
        # Check regularity
        is_regular = (2.5 <= median_gap <= 3.5)
        regularity_score = 1 - min(std_gap / median_gap, 1.0) if median_gap > 0 else 0
        
        confidence = 0.95 if is_regular else 0.7
        
        return {
            'detected': 'quarterly',
            'confidence': confidence,
            'observations': len(dates),
            'date_range': (dates.min(), dates.max()),
            'median_gap_months': float(median_gap),
            'std_gap_months': float(std_gap),
            'is_regular': is_regular,
            'regularity_score': regularity_score
        }
    
    def _analyze_regular_frequency(self, series: pd.Series) -> Dict:
        """Analyze regular datetime frequency"""
        try:
            dates = pd.to_datetime(series, errors='coerce')
            dates = dates.dropna().sort_values()
            
            if len(dates) < 2:
                return {
                    'detected': 'unknown',
                    'confidence': 0.0,
                    'reason': 'Insufficient valid dates'
                }
            
            # Calculate gaps in days
            gaps_days = dates.diff().dt.days.dropna()
            
            if len(gaps_days) == 0:
                return {
                    'detected': 'unknown',
                    'confidence': 0.0,
                    'reason': 'Cannot calculate date gaps'
                }
            
            median_gap = gaps_days.median()
            mean_gap = gaps_days.mean()
            std_gap = gaps_days.std()
            
            # Detect frequency
            detected_freq = 'unknown'
            confidence = 0.0
            
            for expected_gap, freq_name, tolerance in self.FREQUENCY_MAP:
                if abs(median_gap - expected_gap) <= tolerance:
                    detected_freq = freq_name
                    
                    # Calculate confidence based on regularity
                    regularity = 1 - min(std_gap / mean_gap, 1.0) if mean_gap > 0 else 0
                    confidence = regularity * 0.7 + 0.3
                    break
            
            # Check regularity
            is_regular = (std_gap / mean_gap) < 0.2 if mean_gap > 0 else False
            
            # Detect unusual gaps
            gap_threshold = mean_gap + 2 * std_gap
            unusual_gaps = gaps_days[gaps_days > gap_threshold]
            
            return {
                'detected': detected_freq,
                'confidence': float(confidence),
                'observations': len(dates),
                'date_range': (dates.min(), dates.max()),
                'median_gap_days': float(median_gap),
                'mean_gap_days': float(mean_gap),
                'std_gap_days': float(std_gap),
                'is_regular': is_regular,
                'unusual_gaps': len(unusual_gaps),
                'gap_examples': unusual_gaps.head(5).tolist() if len(unusual_gaps) > 0 else []
            }
        
        except Exception as e:
            return {
                'detected': 'unknown',
                'confidence': 0.0,
                'reason': f'Error parsing dates: {str(e)}'
            }
    
    def _extract_features(self):
        """Extract numeric features from each sheet"""
        for sheet_name, df in self.dataframes.items():
            sheet_analysis = self.analysis['sheets'][sheet_name]
            
            # Get date column (if exists)
            date_candidates = sheet_analysis.get('date_candidates', [])
            date_col = date_candidates[0]['column'] if date_candidates else None
            
            # Find numeric columns (excluding date)
            numeric_features = []
            
            for col in df.columns:
                if col == date_col:
                    continue
                
                # Try converting to numeric
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    valid_count = numeric_values.notna().sum()
                    total_count = len(df)
                    
                    if valid_count > 0:
                        numeric_rate = valid_count / total_count
                        
                        feature_info = {
                            'name': col,
                            'dtype': str(df[col].dtype),
                            'valid_count': valid_count,
                            'missing_count': total_count - valid_count,
                            'numeric_rate': numeric_rate,
                            'mean': float(numeric_values.mean()) if numeric_rate > 0 else None,
                            'std': float(numeric_values.std()) if numeric_rate > 0 else None,
                            'min': float(numeric_values.min()) if numeric_rate > 0 else None,
                            'max': float(numeric_values.max()) if numeric_rate > 0 else None,
                            'zeros': int((numeric_values == 0).sum()),
                            'unique_values': int(numeric_values.nunique())
                        }
                        
                        # Check if constant
                        is_constant = feature_info['std'] is not None and feature_info['std'] < 1e-10
                        feature_info['is_constant'] = is_constant
                        
                        numeric_features.append(feature_info)
                
                except Exception:
                    continue
            
            sheet_analysis['numeric_features'] = numeric_features
            sheet_analysis['feature_count'] = len(numeric_features)
            
            print(f"   📊 {sheet_name}: Found {len(numeric_features)} numeric features")
            
            # Print feature names
            if numeric_features:
                feature_names = [f['name'] for f in numeric_features[:10]]
                if len(numeric_features) > 10:
                    print(f"      Features: {', '.join(feature_names)} ... (+{len(numeric_features)-10} more)")
                else:
                    print(f"      Features: {', '.join(feature_names)}")
    
    def _assess_quality(self):
        """Assess data quality for each sheet"""
        for sheet_name, df in self.dataframes.items():
            sheet_analysis = self.analysis['sheets'][sheet_name]
            
            # Calculate quality metrics
            total_cells = df.size
            null_cells = df.isnull().sum().sum()
            
            quality_metrics = {
                'total_cells': total_cells,
                'null_cells': null_cells,
                'null_rate': null_cells / total_cells if total_cells > 0 else 0,
                'completeness': 1 - (null_cells / total_cells) if total_cells > 0 else 0,
            }
            
            # Check for duplicates (if date column exists)
            date_candidates = sheet_analysis.get('date_candidates', [])
            if date_candidates:
                date_col = date_candidates[0]['column']
                duplicate_dates = df.duplicated(subset=[date_col]).sum()
                quality_metrics['duplicate_dates'] = duplicate_dates
            
            # Check for constant features
            numeric_features = sheet_analysis.get('numeric_features', [])
            constant_features = [f['name'] for f in numeric_features if f.get('is_constant', False)]
            quality_metrics['constant_features'] = constant_features
            
            # Calculate overall quality score (0-100)
            score = 100
            
            # Penalties
            score -= min(quality_metrics['null_rate'] * 30, 30)  # Missing data
            score -= min(len(constant_features) * 5, 20)  # Constant features
            
            if 'duplicate_dates' in quality_metrics:
                score -= min(quality_metrics['duplicate_dates'] * 2, 20)  # Duplicates
            
            # Frequency regularity bonus
            freq_info = sheet_analysis.get('frequency', {})
            if freq_info.get('is_regular', False):
                score += 10
            
            quality_metrics['quality_score'] = max(0, min(100, score))
            
            sheet_analysis['quality'] = quality_metrics
            
            print(f"   ✅ {sheet_name}: Quality score = {quality_metrics['quality_score']:.0f}/100")
    
    def _generate_recommendations(self):
        """Generate recommendations for each sheet"""
        for sheet_name, df in self.dataframes.items():
            sheet_analysis = self.analysis['sheets'][sheet_name]
            
            recommendations = []
            warnings_list = []
            
            # Check date column
            date_candidates = sheet_analysis.get('date_candidates', [])
            if not date_candidates:
                warnings_list.append("⚠️ No date column detected - manual specification required")
            elif date_candidates[0]['confidence'] < 0.8:
                warnings_list.append(f"⚠️ Low confidence in date column detection ({date_candidates[0]['confidence']:.0%})")
            
            # Check frequency
            freq_info = sheet_analysis.get('frequency', {})
            if freq_info['detected'] == 'unknown':
                warnings_list.append("⚠️ Could not determine time series frequency")
            elif not freq_info.get('is_regular', False):
                warnings_list.append(f"⚠️ Irregular {freq_info['detected']} spacing detected")
                recommendations.append("💡 Check for missing dates and consider interpolation")
            
            # Check features
            feature_count = sheet_analysis.get('feature_count', 0)
            if feature_count == 0:
                warnings_list.append("❌ No numeric features found")
            elif feature_count < 3:
                warnings_list.append(f"⚠️ Only {feature_count} features - may need more predictors")
            
            # Check quality
            quality = sheet_analysis.get('quality', {})
            quality_score = quality.get('quality_score', 0)
            
            if quality_score < 50:
                warnings_list.append(f"❌ Low quality score ({quality_score:.0f}/100)")
                recommendations.append("💡 Significant data cleaning required before modeling")
            elif quality_score < 75:
                warnings_list.append(f"⚠️ Moderate quality score ({quality_score:.0f}/100)")
                recommendations.append("💡 Some data cleaning recommended")
            
            # Check null rate
            null_rate = quality.get('null_rate', 0)
            if null_rate > 0.3:
                warnings_list.append(f"❌ High missing data rate ({null_rate:.0%})")
                recommendations.append("💡 Consider imputation or feature selection")
            elif null_rate > 0.1:
                warnings_list.append(f"⚠️ Moderate missing data ({null_rate:.0%})")
            
            # Check constant features
            constant_features = quality.get('constant_features', [])
            if constant_features:
                warnings_list.append(f"⚠️ {len(constant_features)} constant features detected")
                recommendations.append("💡 Remove constant features before modeling")
            
            # Check duplicates
            if 'duplicate_dates' in quality and quality['duplicate_dates'] > 0:
                warnings_list.append(f"⚠️ {quality['duplicate_dates']} duplicate dates")
                recommendations.append("💡 Remove or merge duplicate dates")
            
            # Check observation count
            obs_count = sheet_analysis['rows']
            if obs_count < 24:
                warnings_list.append(f"⚠️ Few observations ({obs_count}) - results may be unreliable")
            
            # Usage recommendations based on frequency
            if freq_info['detected'] == 'quarterly':
                recommendations.append("📊 Use interpolation to convert to monthly frequency")
            elif freq_info['detected'] == 'daily':
                recommendations.append("📊 Aggregate to monthly for unemployment nowcasting")
            elif freq_info['detected'] == 'monthly':
                recommendations.append("✅ Ready for direct use in monthly models")
            
            sheet_analysis['warnings'] = warnings_list
            sheet_analysis['recommendations'] = recommendations
    
    def _create_error_report(self) -> Dict:
        """Create error report if analysis failed"""
        return {
            'success': False,
            'errors': self.errors,
            'file_name': self.file_name,
            'file_extension': self.file_ext
        }
    
    def print_report(self):
        """Print comprehensive analysis report"""
        if 'sheets' not in self.analysis:
            print("\n❌ Analysis failed. Errors:")
            for error in self.errors:
                print(f"   - {error}")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 COMPREHENSIVE DATA ANALYSIS REPORT")
        print(f"{'='*70}")
        
        print(f"\n📁 File Information:")
        print(f"   Name: {self.file_name}")
        print(f"   Type: {self.file_ext}")
        
        if 'encoding' in self.analysis:
            print(f"   Encoding: {self.analysis['encoding']}")
            print(f"   Delimiter: '{self.analysis['delimiter']}'")
        
        print(f"\n📄 Sheets/Tables: {len(self.analysis['sheets'])}")
        
        for sheet_name, sheet_info in self.analysis['sheets'].items():
            print(f"\n{'-'*70}")
            print(f"📊 SHEET: {sheet_name}")
            print(f"{'-'*70}")
            
            # Structure
            print(f"\n📐 Structure:")
            print(f"   Rows: {sheet_info['rows']:,}")
            print(f"   Columns: {sheet_info['columns']}")
            print(f"   Memory: {sheet_info['memory_mb']:.2f} MB")
            
            # Date column
            print(f"\n📅 Date Column:")
            date_candidates = sheet_info.get('date_candidates', [])
            if date_candidates:
                primary = date_candidates[0]
                print(f"   Column: '{primary['column']}'")
                print(f"   Detection method: {primary['method']}")
                print(f"   Confidence: {primary['confidence']:.0%}")
            else:
                print(f"   ❌ Not detected")
            
            # Frequency
            print(f"\n🔍 Time Series Frequency:")
            freq_info = sheet_info.get('frequency', {})
            print(f"   Detected: {freq_info['detected'].upper()}")
            print(f"   Confidence: {freq_info['confidence']:.0%}")
            
            if 'observations' in freq_info:
                print(f"   Observations: {freq_info['observations']}")
            
            if 'date_range' in freq_info:
                start, end = freq_info['date_range']
                print(f"   Range: {start.date()} to {end.date()}")
            
            if 'median_gap_days' in freq_info:
                print(f"   Median gap: {freq_info['median_gap_days']:.1f} days")
            
            if 'median_gap_months' in freq_info:
                print(f"   Median gap: {freq_info['median_gap_months']:.2f} months")
            
            if 'is_regular' in freq_info:
                print(f"   Regular spacing: {'Yes ✓' if freq_info['is_regular'] else 'No ✗'}")
            
            # Features
            print(f"\n📊 Numeric Features: {sheet_info['feature_count']}")
            
            features = sheet_info.get('numeric_features', [])
            if features:
                print(f"\n   Top features by completeness:")
                sorted_features = sorted(features, key=lambda x: x['numeric_rate'], reverse=True)
                
                for i, feat in enumerate(sorted_features[:10], 1):
                    status = "✓" if feat['numeric_rate'] > 0.9 else "⚠️"
                    const_flag = " [CONSTANT]" if feat.get('is_constant', False) else ""
                    print(f"   {i:2d}. {feat['name'][:40]:40s} {status} ({feat['numeric_rate']:.0%}){const_flag}")
                
                if len(features) > 10:
                    print(f"   ... and {len(features)-10} more features")
            
            # Quality
            print(f"\n✅ Data Quality:")
            quality = sheet_info.get('quality', {})
            score = quality.get('quality_score', 0)
            
            if score >= 90:
                status = "🌟 EXCELLENT"
            elif score >= 75:
                status = "✅ GOOD"
            elif score >= 50:
                status = "⚠️ FAIR"
            else:
                status = "❌ POOR"
            
            print(f"   Overall Score: {score:.0f}/100 {status}")
            print(f"   Completeness: {quality.get('completeness', 0):.1%}")
            print(f"   Missing cells: {quality.get('null_cells', 0):,} ({quality.get('null_rate', 0):.1%})")
            
            if 'duplicate_dates' in quality:
                print(f"   Duplicate dates: {quality['duplicate_dates']}")
            
            if quality.get('constant_features'):
                print(f"   Constant features: {len(quality['constant_features'])}")
            
            # Warnings
            warnings_list = sheet_info.get('warnings', [])
            if warnings_list:
                print(f"\n⚠️  Warnings ({len(warnings_list)}):")
                for warning in warnings_list:
                    print(f"   {warning}")
            
            # Recommendations
            recommendations = sheet_info.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations ({len(recommendations)}):")
                for rec in recommendations:
                    print(f"   {rec}")
        
        print(f"\n{'='*70}")
        print(f"✅ Analysis Complete")
        print(f"{'='*70}\n")
    
    def get_summary(self) -> Dict:
        """Get concise summary for programmatic use"""
        if 'sheets' not in self.analysis:
            return {'success': False, 'errors': self.errors}
        
        summary = {
            'success': True,
            'file_name': self.file_name,
            'file_type': self.file_ext,
            'sheet_count': len(self.analysis['sheets']),
            'sheets': {}
        }
        
        for sheet_name, sheet_info in self.analysis['sheets'].items():
            freq_info = sheet_info.get('frequency', {})
            quality = sheet_info.get('quality', {})
            
            summary['sheets'][sheet_name] = {
                'rows': sheet_info['rows'],
                'columns': sheet_info['columns'],
                'date_column': sheet_info['date_candidates'][0]['column'] if sheet_info.get('date_candidates') else None,
                'frequency': freq_info.get('detected', 'unknown'),
                'frequency_confidence': freq_info.get('confidence', 0),
                'feature_count': sheet_info.get('feature_count', 0),
                'quality_score': quality.get('quality_score', 0),
                'is_usable': (
                    freq_info.get('confidence', 0) > 0.7 and
                    sheet_info.get('feature_count', 0) > 0 and
                    quality.get('quality_score', 0) > 50
                )
            }
        
        return summary


# =============================================================================
# STREAMLIT INTEGRATION
# =============================================================================

def analyze_uploaded_file_streamlit(uploaded_file, show_full_report: bool = True):
    """
    Streamlit-friendly file analysis
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        show_full_report: Show detailed analysis
    
    Returns:
        Analysis dictionary
    """
    import streamlit as st
    
    # Create analyzer
    analyzer = DataFileAnalyzer(uploaded_file, encoding='utf-8')
    
    # Run analysis
    with st.spinner("🔍 Analyzing file structure..."):
        analysis = analyzer.analyze()
    
    if not analysis.get('success', True):
        st.error("❌ Analysis failed")
        for error in analysis.get('errors', []):
            st.error(f"   - {error}")
        return analysis
    
    # Display results
    if show_full_report:
        _display_streamlit_report(analyzer, analysis)
    else:
        _display_streamlit_summary(analyzer, analysis)
    
    return analysis


def _display_streamlit_summary(analyzer, analysis):
    """Display concise summary in Streamlit"""
    import streamlit as st
    
    st.success(f"✅ File analyzed: **{analyzer.file_name}**")
    
    for sheet_name, sheet_info in analysis['sheets'].items():
        with st.expander(f"📊 {sheet_name}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", f"{sheet_info['rows']:,}")
            
            with col2:
                st.metric("Features", sheet_info.get('feature_count', 0))
            
            with col3:
                freq = sheet_info.get('frequency', {})
                st.metric("Frequency", freq.get('detected', 'unknown').upper())
            
            with col4:
                quality = sheet_info.get('quality', {})
                score = quality.get('quality_score', 0)
                st.metric("Quality", f"{score:.0f}/100")


def _display_streamlit_report(analyzer, analysis):
    """Display full report in Streamlit"""
    import streamlit as st
    
    st.markdown("### 📊 Analysis Report")
    
    # File info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**File:** {analyzer.file_name}")
    with col2:
        st.info(f"**Type:** {analyzer.file_ext}")
    
    # Sheets
    for sheet_name, sheet_info in analysis['sheets'].items():
        st.markdown(f"#### 📄 Sheet: {sheet_name}")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Rows", f"{sheet_info['rows']:,}")
        
        with col2:
            st.metric("📊 Columns", sheet_info['columns'])
        
        with col3:
            st.metric("🎯 Features", sheet_info.get('feature_count', 0))
        
        with col4:
            quality = sheet_info.get('quality', {})
            score = quality.get('quality_score', 0)
            st.metric("✅ Quality", f"{score:.0f}/100")
        
        # Date & Frequency
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📅 Date Column:**")
            date_candidates = sheet_info.get('date_candidates', [])
            if date_candidates:
                primary = date_candidates[0]
                st.write(f"✓ `{primary['column']}` ({primary['confidence']:.0%})")
            else:
                st.write("❌ Not detected")
        
        with col2:
            st.markdown("**🔍 Frequency:**")
            freq = sheet_info.get('frequency', {})
            st.write(f"**{freq.get('detected', 'unknown').upper()}** ({freq.get('confidence', 0):.0%})")
        
        # Features
        features = sheet_info.get('numeric_features', [])
        if features:
            with st.expander(f"📊 View {len(features)} Features", expanded=False):
                feat_df = pd.DataFrame(features)
                st.dataframe(
                    feat_df[['name', 'numeric_rate', 'missing_count', 'mean', 'std']],
                    use_container_width=True
                )
        
        # Warnings & Recommendations
        warnings_list = sheet_info.get('warnings', [])
        recommendations = sheet_info.get('recommendations', [])
        
        if warnings_list:
            st.warning("**⚠️ Warnings:**\n\n" + "\n".join(f"- {w}" for w in warnings_list))
        
        if recommendations:
            st.info("**💡 Recommendations:**\n\n" + "\n".join(f"- {r}" for r in recommendations))
        
        st.markdown("---")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Analyze CSV file
    print("\n" + "="*70)
    print("EXAMPLE 1: Quarterly CSV File")
    print("="*70)
    
    # Create sample quarterly data
    sample_data = pd.DataFrame({
        'quarter': [f'{year}-Q{q}' for year in range(2018, 2024) for q in range(1, 5)],
        'unemployment_total': np.random.uniform(6, 10, 24),
        'unemployment_male': np.random.uniform(5.5, 9.5, 24),
        'unemployment_female': np.random.uniform(6.5, 10.5, 24),
        'unemployment_youth': np.random.uniform(18, 30, 24),
    })
    
    # Save to CSV
    sample_data.to_csv('sample_quarterly.csv', index=False)
    
    # Analyze
    analyzer = DataFileAnalyzer('sample_quarterly.csv')
    analysis = analyzer.analyze()
    analyzer.print_report()
    
    # Get summary
    summary = analyzer.get_summary()
    print("\n📋 Summary for programmatic use:")
    print(f"   Success: {summary['success']}")
    print(f"   Sheets: {summary['sheet_count']}")
    for sheet_name, sheet_data in summary['sheets'].items():
        print(f"   {sheet_name}:")
        print(f"      - Frequency: {sheet_data['frequency']}")
        print(f"      - Features: {sheet_data['feature_count']}")
        print(f"      - Usable: {sheet_data['is_usable']}")
