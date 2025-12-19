"""
Export and Report Generation
CSV, Excel, PDF reports, ZIP packaging
Production-ready export functionality
"""

import os
import io
import json
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG


class DataExporter:
    """
    Export data to various formats
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or CONFIG.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_predictions(self,
                          dates: pd.Series,
                          actual: np.ndarray,
                          predictions: Dict[str, np.ndarray],
                          filename: str = "predictions.csv") -> str:
        """
        Export predictions to CSV
        
        Args:
            dates: Date series
            actual: Actual values
            predictions: Dict of {model_name: predictions}
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        # Build dataframe
        df = pd.DataFrame({
            'date': dates,
            'actual': actual
        })
        
        # Add predictions
        for model_name, pred in predictions.items():
            # Clean model name for column
            col_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
            df[f'pred_{col_name}'] = pred
            
            # Add errors
            df[f'error_{col_name}'] = actual - pred
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_metrics(self,
                      metrics: Dict[str, Dict[str, float]],
                      filename: str = "metrics.csv") -> str:
        """
        Export metrics comparison to CSV
        
        Args:
            metrics: Dict of {model_name: {metric: value}}
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        # Convert to dataframe
        df = pd.DataFrame(metrics).T
        df.index.name = 'model'
        df.reset_index(inplace=True)
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_backtest_results(self,
                               backtest_results: List,
                               filename: str = "backtest_results.csv") -> str:
        """
        Export backtest results to CSV
        
        Args:
            backtest_results: List of BacktestResult objects
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        records = []
        
        for result in backtest_results:
            record = {
                'model_name': result.model_name,
                'split_id': result.split_id,
                'n_train': result.n_train,
                'n_test': result.n_test
            }
            
            # Add metrics
            record.update(result.metrics)
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_statistical_tests(self,
                                 test_results: Dict[str, Any],
                                 filename: str = "statistical_tests.csv") -> str:
        """
        Export statistical test results to CSV
        
        Args:
            test_results: Dict of test results
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        records = []
        
        for test_name, result in test_results.items():
            record = {
                'test': test_name,
                'statistic': result.get('statistic', np.nan),
                'p_value': result.get('p_value', np.nan),
                'is_significant': result.get('is_significant', False),
                'interpretation': result.get('interpretation', '')
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_feature_importance(self,
                                  importance: Dict[str, float],
                                  filename: str = "feature_importance.csv") -> str:
        """
        Export feature importance to CSV
        
        Args:
            importance: Dict of {feature: importance}
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
        df = df.sort_values('importance', key=abs, ascending=False)
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_to_excel(self,
                       data_dict: Dict[str, pd.DataFrame],
                       filename: str = "nowcast_results.xlsx") -> str:
        """
        Export multiple dataframes to Excel with sheets
        
        Args:
            data_dict: Dict of {sheet_name: dataframe}
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Truncate sheet name if needed (Excel limit: 31 chars)
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return filepath
    
    def export_json(self,
                   data: Dict,
                   filename: str = "results.json") -> str:
        """
        Export data to JSON
        
        Args:
            data: Dictionary to export
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types
        data_clean = self._clean_for_json(data)
        
        with open(filepath, 'w') as f:
            json.dump(data_clean, f, indent=2)
        
        return filepath
    
    def _clean_for_json(self, obj):
        """Recursively clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj


class FigureExporter:
    """
    Export Plotly figures to various formats
    """
    
    def __init__(self, figures_dir: Optional[str] = None):
        self.figures_dir = figures_dir or CONFIG.FIGURES_DIR
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def export_figure(self,
                     fig,
                     filename: str,
                     formats: List[str] = ['png', 'html']) -> Dict[str, str]:
        """
        Export Plotly figure to multiple formats
        
        Args:
            fig: Plotly figure
            filename: Base filename (without extension)
            formats: List of formats ('png', 'html', 'svg', 'pdf')
        
        Returns:
            Dict of {format: filepath}
        """
        filepaths = {}
        
        for fmt in formats:
            if fmt == 'html':
                filepath = os.path.join(self.figures_dir, f"{filename}.html")
                fig.write_html(filepath)
                filepaths['html'] = filepath
            
            elif fmt == 'png':
                try:
                    filepath = os.path.join(self.figures_dir, f"{filename}.png")
                    fig.write_image(filepath, width=1200, height=800, scale=2)
                    filepaths['png'] = filepath
                except Exception as e:
                    print(f"PNG export failed: {e}")
            
            elif fmt == 'svg':
                try:
                    filepath = os.path.join(self.figures_dir, f"{filename}.svg")
                    fig.write_image(filepath)
                    filepaths['svg'] = filepath
                except Exception as e:
                    print(f"SVG export failed: {e}")
            
            elif fmt == 'pdf':
                try:
                    filepath = os.path.join(self.figures_dir, f"{filename}.pdf")
                    fig.write_image(filepath)
                    filepaths['pdf'] = filepath
                except Exception as e:
                    print(f"PDF export failed: {e}")
        
        return filepaths
    
    def export_all_figures(self,
                          figures: Dict[str, Any],
                          formats: List[str] = ['png', 'html']) -> Dict[str, Dict[str, str]]:
        """
        Export multiple figures
        
        Args:
            figures: Dict of {name: figure}
            formats: List of formats
        
        Returns:
            Dict of {name: {format: filepath}}
        """
        all_paths = {}
        
        for name, fig in figures.items():
            paths = self.export_figure(fig, name, formats)
            all_paths[name] = paths
        
        return all_paths


class ReportGenerator:
    """
    Generate comprehensive reports
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or CONFIG.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_summary_report(self,
                               project_info: Dict,
                               data_info: Dict,
                               metrics: Dict[str, Dict],
                               test_results: Dict,
                               filename: str = "summary_report.txt") -> str:
        """
        Generate text summary report
        
        Args:
            project_info: Project metadata
            data_info: Data information
            metrics: Model metrics
            test_results: Statistical test results
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("NOWCASTING PLATFORM - SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Project info
        lines.append("-" * 80)
        lines.append("PROJECT INFORMATION")
        lines.append("-" * 80)
        for key, value in project_info.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        
        # Data info
        lines.append("-" * 80)
        lines.append("DATA INFORMATION")
        lines.append("-" * 80)
        for key, value in data_info.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        
        # Metrics
        lines.append("-" * 80)
        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 80)
        
        # Find best model
        if metrics:
            rmse_scores = {model: m.get('rmse', np.inf) for model, m in metrics.items()}
            best_model = min(rmse_scores, key=rmse_scores.get)
            
            lines.append(f"Best Model (by RMSE): {best_model}")
            lines.append("")
            
            # Table header
            lines.append(f"{'Model':<30} {'RMSE':<12} {'MAE':<12} {'Dir Acc':<12}")
            lines.append("-" * 66)
            
            for model, m in metrics.items():
                rmse = m.get('rmse', np.nan)
                mae = m.get('mae', np.nan)
                dir_acc = m.get('direction_accuracy', np.nan)
                
                marker = " ✓" if model == best_model else ""
                lines.append(f"{model:<30} {rmse:<12.4f} {mae:<12.4f} {dir_acc:<12.1f}{marker}")
        
        lines.append("")
        
        # Statistical tests
        if test_results:
            lines.append("-" * 80)
            lines.append("STATISTICAL TESTS")
            lines.append("-" * 80)
            
            for test_name, result in test_results.items():
                stat = result.get('statistic', np.nan)
                p_val = result.get('p_value', np.nan)
                sig = result.get('is_significant', False)
                
                sig_marker = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                
                lines.append(f"{test_name}:")
                lines.append(f"  Statistic: {stat:.4f}")
                lines.append(f"  P-value: {p_val:.4f} {sig_marker}")
                lines.append(f"  Significant: {'Yes' if sig else 'No'}")
                lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Write to file
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def generate_markdown_report(self,
                                project_info: Dict,
                                data_info: Dict,
                                metrics: Dict[str, Dict],
                                test_results: Dict,
                                figures: Dict[str, str],
                                filename: str = "report.md") -> str:
        """
        Generate Markdown report with figures
        
        Args:
            project_info: Project metadata
            data_info: Data information
            metrics: Model metrics
            test_results: Statistical test results
            figures: Dict of {figure_name: filepath}
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        lines = []
        
        # Header
        lines.append("# Nowcasting Platform Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Project info
        lines.append("## Project Information")
        lines.append("")
        for key, value in project_info.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
        
        # Data info
        lines.append("## Data Information")
        lines.append("")
        for key, value in data_info.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
        
        # Metrics table
        lines.append("## Model Performance")
        lines.append("")
        
        if metrics:
            lines.append("| Model | RMSE | MAE | MAPE | Dir Acc |")
            lines.append("|-------|------|-----|------|---------|")
            
            for model, m in metrics.items():
                rmse = m.get('rmse', np.nan)
                mae = m.get('mae', np.nan)
                mape = m.get('mape', np.nan)
                dir_acc = m.get('direction_accuracy', np.nan)
                
                lines.append(f"| {model} | {rmse:.4f} | {mae:.4f} | {mape:.2f} | {dir_acc:.1f}% |")
        
        lines.append("")
        
        # Statistical tests
        if test_results:
            lines.append("## Statistical Tests")
            lines.append("")
            
            for test_name, result in test_results.items():
                stat = result.get('statistic', np.nan)
                p_val = result.get('p_value', np.nan)
                sig = result.get('is_significant', False)
                
                lines.append(f"### {test_name}")
                lines.append("")
                lines.append(f"- **Statistic:** {stat:.4f}")
                lines.append(f"- **P-value:** {p_val:.4f}")
                lines.append(f"- **Significant:** {'✅ Yes' if sig else '❌ No'}")
                lines.append("")
        
        # Figures
        if figures:
            lines.append("## Visualizations")
            lines.append("")
            
            for fig_name, fig_path in figures.items():
                lines.append(f"### {fig_name}")
                lines.append("")
                
                # Use relative path
                rel_path = os.path.relpath(fig_path, os.path.dirname(os.path.join(self.output_dir, filename)))
                lines.append(f"![{fig_name}]({rel_path})")
                lines.append("")
        
        # Write to file
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath


class PackageExporter:
    """
    Create complete export packages
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or CONFIG.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_zip_package(self,
                          files: List[str],
                          package_name: str = "nowcast_results",
                          include_readme: bool = True) -> str:
        """
        Create ZIP package with all results
        
        Args:
            files: List of file paths to include
            package_name: Name of ZIP file (without .zip)
            include_readme: Include README file
        
        Returns:
            Path to ZIP file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"{package_name}_{timestamp}.zip"
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files
            for filepath in files:
                if os.path.exists(filepath):
                    arcname = os.path.basename(filepath)
                    zipf.write(filepath, arcname)
            
            # Add README
            if include_readme:
                readme_content = self._generate_readme()
                zipf.writestr('README.txt', readme_content)
        
        return zip_path
    
    def create_complete_package(self,
                               predictions_df: pd.DataFrame,
                               metrics: Dict[str, Dict],
                               test_results: Dict,
                               figures: Dict[str, Any],
                               backtest_results: Optional[List] = None,
                               feature_importance: Optional[Dict] = None) -> str:
        """
        Create complete export package with all results
        
        Args:
            predictions_df: Predictions dataframe
            metrics: Model metrics
            test_results: Statistical test results
            figures: Dict of {name: figure}
            backtest_results: Optional backtest results
            feature_importance: Optional feature importance
        
        Returns:
            Path to ZIP package
        """
        files_to_zip = []
        
        # Export data
        data_exporter = DataExporter(self.output_dir)
        
        # Predictions
        pred_file = data_exporter.export_predictions(
            predictions_df['date'],
            predictions_df['actual'].values,
            {col.replace('pred_', ''): predictions_df[col].values 
             for col in predictions_df.columns if col.startswith('pred_')},
            filename='predictions.csv'
        )
        files_to_zip.append(pred_file)
        
        # Metrics
        metrics_file = data_exporter.export_metrics(metrics, filename='metrics.csv')
        files_to_zip.append(metrics_file)
        
        # Statistical tests
        tests_file = data_exporter.export_statistical_tests(test_results, filename='statistical_tests.csv')
        files_to_zip.append(tests_file)
        
        # Backtest results
        if backtest_results:
            backtest_file = data_exporter.export_backtest_results(backtest_results, filename='backtest_results.csv')
            files_to_zip.append(backtest_file)
        
        # Feature importance
        if feature_importance:
            importance_file = data_exporter.export_feature_importance(feature_importance, filename='feature_importance.csv')
            files_to_zip.append(importance_file)
        
        # Export figures
        figure_exporter = FigureExporter(self.figures_dir)
        figure_paths = figure_exporter.export_all_figures(figures, formats=['png', 'html'])
        
        for fig_name, paths in figure_paths.items():
            files_to_zip.extend(paths.values())
        
        # Generate reports
        report_gen = ReportGenerator(self.output_dir)
        
        project_info = {
            'Platform': 'Nowcasting Platform v1.0',
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        data_info = {
            'Observations': len(predictions_df),
            'Models': len(metrics)
        }
        
        summary_file = report_gen.generate_summary_report(
            project_info, data_info, metrics, test_results,
            filename='summary_report.txt'
        )
        files_to_zip.append(summary_file)
        
        # Create ZIP
        zip_path = self.create_zip_package(files_to_zip, package_name='nowcast_complete')
        
        return zip_path
    
    def _generate_readme(self) -> str:
        """Generate README content for package"""
        lines = [
            "NOWCASTING PLATFORM - RESULTS PACKAGE",
            "=" * 80,
            "",
            "This package contains the complete results from your nowcasting analysis.",
            "",
            "CONTENTS:",
            "-" * 80,
            "",
            "1. predictions.csv",
            "   - Time series of actual values and model predictions",
            "   - Includes forecast errors for each model",
            "",
            "2. metrics.csv",
            "   - Performance metrics for all models",
            "   - RMSE, MAE, MAPE, Direction Accuracy, etc.",
            "",
            "3. statistical_tests.csv",
            "   - Results from Diebold-Mariano, Clark-West tests",
            "   - P-values and significance indicators",
            "",
            "4. backtest_results.csv (if applicable)",
            "   - Rolling-origin backtest performance",
            "   - Performance across multiple train/test splits",
            "",
            "5. feature_importance.csv (if applicable)",
            "   - Feature importance scores",
            "   - Sorted by absolute importance",
            "",
            "6. Figures/",
            "   - PNG and HTML versions of all charts",
            "   - predictions_vs_actual.png",
            "   - error_distribution.png",
            "   - metrics_comparison.png",
            "   - etc.",
            "",
            "7. summary_report.txt",
            "   - Human-readable summary of results",
            "   - Best model identification",
            "   - Key findings",
            "",
            "=" * 80,
            "For questions or support, please refer to the platform documentation.",
            ""
        ]
        
        return '\n'.join(lines)


class StreamlitDownloader:
    """
    Helper class for Streamlit downloads
    """
    
    @staticmethod
    def prepare_csv_download(df: pd.DataFrame) -> bytes:
        """
        Prepare CSV for Streamlit download
        
        Args:
            df: DataFrame to export
        
        Returns:
            CSV as bytes
        """
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def prepare_excel_download(data_dict: Dict[str, pd.DataFrame]) -> bytes:
        """
        Prepare Excel for Streamlit download
        
        Args:
            data_dict: Dict of {sheet_name: dataframe}
        
        Returns:
            Excel as bytes
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                sheet_name = sheet_name[:31]  # Excel limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output.read()
    
    @staticmethod
    def prepare_json_download(data: Dict) -> bytes:
        """
        Prepare JSON for Streamlit download
        
        Args:
            data: Dictionary to export
        
        Returns:
            JSON as bytes
        """
        # Clean for JSON
        data_clean = DataExporter(None)._clean_for_json(data)
        return json.dumps(data_clean, indent=2).encode('utf-8')
    
    @staticmethod
    def prepare_zip_download(files_dict: Dict[str, bytes]) -> bytes:
        """
        Prepare ZIP for Streamlit download
        
        Args:
            files_dict: Dict of {filename: file_bytes}
        
        Returns:
            ZIP as bytes
        """
        output = io.BytesIO()
        
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename, file_bytes in files_dict.items():
                zipf.writestr(filename, file_bytes)
        
        output.seek(0)
        return output.read()
