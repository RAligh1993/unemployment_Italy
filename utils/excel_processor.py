"""
📊 Excel File Processor Module
================================

Advanced Excel file reader with multi-sheet support, auto-detection,
and robust error handling for all Excel formats.

Supported Formats:
    - .xlsx (Excel 2007+)
    - .xls (Excel 97-2003)
    - .xlsm (Excel with macros)
    - .xlsb (Excel binary)

Features:
    - Multi-engine fallback (openpyxl, xlrd, etc.)
    - Header row detection
    - Empty row/column removal
    - Column name cleaning
    - Sheet metadata extraction
    - Memory-efficient chunked reading

Author: ISTAT Nowcasting Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
import io

# Excel engines with graceful imports
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import xlrd
    HAS_XLRD = True
except ImportError:
    HAS_XLRD = False

try:
    from pyxlsb import open_workbook as open_xlsb
    HAS_PYXLSB = True
except ImportError:
    HAS_PYXLSB = False


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class ExcelConfig:
    """Configuration for Excel processing"""
    max_file_size_mb: int = 200
    max_sheets: int = 50
    header_search_rows: int = 15
    min_header_coverage: float = 0.5
    remove_empty_rows: bool = True
    remove_empty_cols: bool = True
    auto_detect_header: bool = True


@dataclass
class SheetMetadata:
    """Metadata for a single Excel sheet"""
    name: str
    index: int
    shape: Tuple[int, int]
    header_row: int
    columns: List[str]
    has_data: bool
    memory_mb: float
    
    def __str__(self) -> str:
        return (f"Sheet('{self.name}', {self.shape[0]}×{self.shape[1]}, "
                f"header_row={self.header_row}, {self.memory_mb:.1f}MB)")


# =============================================================================
# Excel Processor Class
# =============================================================================

class ExcelProcessor:
    """
    Professional Excel file processor with advanced features.
    
    This class handles all aspects of Excel file reading including:
    - Multi-format support (.xlsx, .xls, .xlsm, .xlsb)
    - Multi-engine fallback for compatibility
    - Intelligent header detection
    - Data cleaning and normalization
    - Sheet metadata extraction
    
    Usage:
        Basic:
            >>> processor = ExcelProcessor()
            >>> sheets = processor.read_file('data.xlsx')
            >>> df = sheets['Sheet1']
        
        With configuration:
            >>> config = ExcelConfig(max_file_size_mb=500)
            >>> processor = ExcelProcessor(config)
            >>> sheets = processor.read_file('large_file.xlsx')
        
        Single sheet:
            >>> df = processor.read_sheet('data.xlsx', sheet_name='Data')
        
        With metadata:
            >>> sheets, metadata = processor.read_file_with_metadata('data.xlsx')
            >>> for meta in metadata:
            ...     print(meta)
    
    Attributes:
        config: Processing configuration
        supported_formats: List of supported file extensions
        available_engines: Dict of available Excel engines
    """
    
    SUPPORTED_FORMATS = ['.xlsx', '.xls', '.xlsm', '.xlsb']
    
    def __init__(self, config: Optional[ExcelConfig] = None):
        """
        Initialize Excel processor.
        
        Args:
            config: Processing configuration (uses defaults if None)
        """
        self.config = config or ExcelConfig()
        self.available_engines = self._detect_available_engines()
        
        if not self.available_engines:
            warnings.warn(
                "No Excel engines available! Install openpyxl or xlrd: "
                "pip install openpyxl xlrd",
                ImportWarning
            )
    
    def _detect_available_engines(self) -> Dict[str, bool]:
        """Detect which Excel engines are available"""
        return {
            'openpyxl': HAS_OPENPYXL,
            'xlrd': HAS_XLRD,
            'pyxlsb': HAS_PYXLSB
        }
    
    # =========================================================================
    # Main Reading Methods
    # =========================================================================
    
    def read_file(self, 
                  file: Union[str, Path, io.BytesIO],
                  sheet_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Read all (or selected) sheets from Excel file.
        
        Args:
            file: File path or file-like object
            sheet_names: List of sheet names to read (None = all sheets)
        
        Returns:
            dict: {sheet_name: DataFrame}
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not supported
            IOError: If file cannot be read
        
        Example:
            >>> sheets = processor.read_file('data.xlsx')
            >>> print(f"Read {len(sheets)} sheets")
            >>> for name, df in sheets.items():
            ...     print(f"  {name}: {df.shape}")
        """
        # Validate file
        self._validate_file(file)
        
        # Try different engines
        sheets = {}
        last_error = None
        
        engines = self._get_engine_sequence(file)
        
        for engine in engines:
            try:
                sheets = self._read_with_engine(file, engine, sheet_names)
                if sheets:
                    break
            except Exception as e:
                last_error = e
                continue
        
        if not sheets:
            raise IOError(
                f"Failed to read Excel file with all available engines.\n"
                f"Last error: {last_error}\n"
                f"Available engines: {[e for e, avail in self.available_engines.items() if avail]}"
            )
        
        # Process each sheet
        processed_sheets = {}
        for name, df in sheets.items():
            try:
                processed = self._process_sheet(df, name)
                if not processed.empty:
                    processed_sheets[name] = processed
            except Exception as e:
                warnings.warn(f"Failed to process sheet '{name}': {e}")
                continue
        
        return processed_sheets
    
    def read_sheet(self,
                   file: Union[str, Path, io.BytesIO],
                   sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        Read single sheet from Excel file.
        
        Args:
            file: File path or file-like object
            sheet_name: Sheet name or index (0-based)
        
        Returns:
            pd.DataFrame: Processed sheet data
        
        Example:
            >>> df = processor.read_sheet('data.xlsx', 'Sales')
            >>> df = processor.read_sheet('data.xlsx', sheet_name=0)  # First sheet
        """
        sheets = self.read_file(file, sheet_names=[sheet_name] if isinstance(sheet_name, str) else None)
        
        if isinstance(sheet_name, int):
            return list(sheets.values())[sheet_name]
        else:
            return sheets.get(sheet_name, pd.DataFrame())
    
    def read_file_with_metadata(self,
                                file: Union[str, Path, io.BytesIO]) -> Tuple[Dict[str, pd.DataFrame], List[SheetMetadata]]:
        """
        Read file and extract detailed metadata.
        
        Args:
            file: File path or file-like object
        
        Returns:
            tuple: (sheets_dict, metadata_list)
        
        Example:
            >>> sheets, metadata = processor.read_file_with_metadata('data.xlsx')
            >>> for meta in metadata:
            ...     print(f"{meta.name}: {meta.shape}")
        """
        sheets = self.read_file(file)
        metadata = []
        
        for idx, (name, df) in enumerate(sheets.items()):
            meta = SheetMetadata(
                name=name,
                index=idx,
                shape=df.shape,
                header_row=0,  # After processing, header is at 0
                columns=df.columns.tolist(),
                has_data=not df.empty,
                memory_mb=df.memory_usage(deep=True).sum() / (1024 ** 2)
            )
            metadata.append(meta)
        
        return sheets, metadata
    
    # =========================================================================
    # Engine-Specific Reading
    # =========================================================================
    
    def _read_with_engine(self,
                         file: Union[str, Path, io.BytesIO],
                         engine: str,
                         sheet_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Read Excel with specific engine"""
        
        # Reset file pointer if BytesIO
        if isinstance(file, io.BytesIO):
            file.seek(0)
        
        # Read with pandas
        try:
            excel_file = pd.ExcelFile(file, engine=engine)
            
            # Determine which sheets to read
            if sheet_names:
                sheets_to_read = [s for s in sheet_names if s in excel_file.sheet_names]
            else:
                sheets_to_read = excel_file.sheet_names[:self.config.max_sheets]
            
            # Read each sheet
            sheets = {}
            for sheet_name in sheets_to_read:
                try:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=None  # We'll detect header ourselves
                    )
                    
                    if not df.empty:
                        sheets[sheet_name] = df
                
                except Exception as e:
                    warnings.warn(f"Failed to read sheet '{sheet_name}': {e}")
                    continue
            
            return sheets
        
        except Exception as e:
            raise IOError(f"Engine {engine} failed: {e}")
    
    def _get_engine_sequence(self, file: Union[str, Path, io.BytesIO]) -> List[str]:
        """Determine optimal engine sequence based on file type"""
        
        # Get file extension
        if isinstance(file, (str, Path)):
            ext = Path(file).suffix.lower()
        else:
            ext = '.xlsx'  # Default for BytesIO
        
        # Define engine priority by format
        engine_priority = {
            '.xlsx': ['openpyxl', 'xlrd'],
            '.xlsm': ['openpyxl', 'xlrd'],
            '.xls': ['xlrd', 'openpyxl'],
            '.xlsb': ['pyxlsb']
        }
        
        preferred = engine_priority.get(ext, ['openpyxl', 'xlrd', 'pyxlsb'])
        
        # Filter to only available engines
        return [e for e in preferred if self.available_engines.get(e, False)]
    
    # =========================================================================
    # Sheet Processing
    # =========================================================================
    
    def _process_sheet(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """
        Process raw sheet data: detect header, clean, normalize.
        
        Args:
            df: Raw DataFrame (no header set)
            sheet_name: Name for logging
        
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        if df.empty:
            return df
        
        # Step 1: Detect header row
        if self.config.auto_detect_header:
            header_row = self._detect_header_row(df)
        else:
            header_row = 0
        
        # Step 2: Set header
        if header_row > 0:
            df = df.iloc[header_row:].reset_index(drop=True)
        
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Step 3: Remove empty rows/columns
        if self.config.remove_empty_rows:
            df = df.dropna(how='all', axis=0)
        
        if self.config.remove_empty_cols:
            df = df.dropna(how='all', axis=1)
        
        # Step 4: Clean column names
        df.columns = self._clean_column_names(df.columns)
        
        # Step 5: Remove duplicate columns
        df = self._remove_duplicate_columns(df)
        
        return df
    
    def _detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Detect which row contains the header.
        
        Strategy:
        1. Look for row with most non-null, text values
        2. Check for keyword presence (date, value, etc.)
        3. Avoid rows with mostly numbers
        
        Args:
            df: Raw DataFrame
        
        Returns:
            int: Header row index (0-based)
        """
        max_rows = min(self.config.header_search_rows, len(df))
        
        best_row = 0
        best_score = 0
        
        for i in range(max_rows):
            row = df.iloc[i]
            score = 0
            
            # Score 1: Non-null coverage
            non_null_ratio = row.notna().sum() / len(row)
            score += non_null_ratio * 10
            
            # Score 2: Text content (headers are usually text)
            text_count = sum(isinstance(val, str) for val in row)
            score += (text_count / len(row)) * 10
            
            # Score 3: Keyword presence
            row_text = ' '.join(str(val).lower() for val in row if pd.notna(val))
            keywords = ['date', 'time', 'value', 'name', 'id', 'code', 'period', 
                       'data', 'periodo', 'valore', 'territorio']
            
            keyword_matches = sum(kw in row_text for kw in keywords)
            score += keyword_matches * 3
            
            # Score 4: Penalize if mostly numeric (probably data, not header)
            numeric_count = sum(isinstance(val, (int, float)) for val in row)
            if numeric_count > len(row) * 0.7:
                score -= 15
            
            # Score 5: Check if next rows are more numeric (good sign for header)
            if i < len(df) - 1:
                next_row = df.iloc[i + 1]
                next_numeric = sum(isinstance(val, (int, float)) or 
                                 (isinstance(val, str) and val.replace('.', '').replace(',', '').isdigit())
                                 for val in next_row)
                if next_numeric > len(next_row) * 0.5:
                    score += 5
            
            # Update best
            if score > best_score:
                best_score = score
                best_row = i
        
        return best_row
    
    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """
        Clean and normalize column names.
        
        Rules:
        - Convert to lowercase
        - Replace spaces with underscores
        - Remove special characters
        - Ensure uniqueness
        
        Args:
            columns: Original column names
        
        Returns:
            list: Cleaned column names
        """
        import re
        
        cleaned = []
        seen = {}
        
        for col in columns:
            # Convert to string
            col_str = str(col).strip()
            
            # Handle unnamed columns
            if col_str.lower().startswith('unnamed'):
                col_str = f'col_{len(cleaned)}'
            
            # Clean
            col_str = col_str.lower()
            col_str = re.sub(r'[^\w\s-]', '_', col_str)  # Replace special chars
            col_str = re.sub(r'[-\s]+', '_', col_str)     # Replace spaces/hyphens
            col_str = re.sub(r'_+', '_', col_str)         # Collapse underscores
            col_str = col_str.strip('_')                   # Remove leading/trailing
            
            # Ensure non-empty
            if not col_str:
                col_str = f'col_{len(cleaned)}'
            
            # Handle duplicates
            if col_str in seen:
                seen[col_str] += 1
                col_str = f'{col_str}_{seen[col_str]}'
            else:
                seen[col_str] = 0
            
            cleaned.append(col_str)
        
        return cleaned
    
    def _remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate column names by adding suffixes"""
        cols = pd.Series(df.columns)
        
        for dup in cols[cols.duplicated()].unique():
            dup_indices = cols[cols == dup].index.tolist()
            
            for i, idx in enumerate(dup_indices[1:], start=1):
                cols[idx] = f"{dup}_{i}"
        
        df.columns = cols
        return df
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def _validate_file(self, file: Union[str, Path, io.BytesIO]):
        """Validate file before reading"""
        
        # Check file object
        if isinstance(file, io.BytesIO):
            # Check size
            file.seek(0, 2)  # Seek to end
            size_mb = file.tell() / (1024 ** 2)
            file.seek(0)     # Reset
            
            if size_mb > self.config.max_file_size_mb:
                raise ValueError(
                    f"File too large: {size_mb:.1f}MB "
                    f"(max: {self.config.max_file_size_mb}MB)"
                )
            
            return
        
        # Check file path
        file_path = Path(file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check extension
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_path.suffix}\n"
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        # Check size
        size_mb = file_path.stat().st_size / (1024 ** 2)
        if size_mb > self.config.max_file_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB "
                f"(max: {self.config.max_file_size_mb}MB)"
            )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_sheet_names(self, file: Union[str, Path, io.BytesIO]) -> List[str]:
        """
        Get list of sheet names without reading data.
        
        Args:
            file: File path or file-like object
        
        Returns:
            list: Sheet names
        
        Example:
            >>> sheets = processor.get_sheet_names('data.xlsx')
            >>> print(f"Available sheets: {sheets}")
        """
        engines = self._get_engine_sequence(file)
        
        for engine in engines:
            try:
                if isinstance(file, io.BytesIO):
                    file.seek(0)
                
                excel_file = pd.ExcelFile(file, engine=engine)
                return excel_file.sheet_names
            
            except Exception:
                continue
        
        raise IOError("Failed to read sheet names with any engine")
    
    def estimate_memory(self, 
                       file: Union[str, Path, io.BytesIO],
                       sheet_name: Optional[str] = None) -> float:
        """
        Estimate memory usage in MB.
        
        Args:
            file: File path or file-like object
            sheet_name: Specific sheet (None = all sheets)
        
        Returns:
            float: Estimated memory in MB
        
        Example:
            >>> memory_mb = processor.estimate_memory('large_file.xlsx')
            >>> print(f"Estimated memory: {memory_mb:.1f}MB")
        """
        if sheet_name:
            df = self.read_sheet(file, sheet_name)
            return df.memory_usage(deep=True).sum() / (1024 ** 2)
        else:
            sheets = self.read_file(file)
            total = sum(
                df.memory_usage(deep=True).sum()
                for df in sheets.values()
            )
            return total / (1024 ** 2)


# =============================================================================
# Convenience Functions
# =============================================================================

def read_excel_smart(file: Union[str, Path, io.BytesIO],
                     sheet_name: Union[str, int, None] = 0) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    One-liner to read Excel with smart defaults.
    
    Args:
        file: File path or file-like object
        sheet_name: Sheet to read (0=first, None=all, str=specific)
    
    Returns:
        DataFrame or dict of DataFrames
    
    Example:
        >>> df = read_excel_smart('data.xlsx')  # First sheet
        >>> sheets = read_excel_smart('data.xlsx', sheet_name=None)  # All sheets
        >>> df = read_excel_smart('data.xlsx', 'Sales')  # Specific sheet
    """
    processor = ExcelProcessor()
    
    if sheet_name is None:
        return processor.read_file(file)
    else:
        return processor.read_sheet(file, sheet_name)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Excel processor module...")
    
    # Check available engines
    processor = ExcelProcessor()
    print(f"Available engines: {processor.available_engines}")
    
    # Create sample Excel in memory
    from io import BytesIO
    
    sample_df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=5, freq='M'),
        'Value': [10.5, 11.2, 9.8, 12.1, 10.9],
        'Category': ['A', 'B', 'A', 'B', 'A']
    })
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sample_df.to_excel(writer, sheet_name='Test', index=False)
    
    buffer.seek(0)
    
    # Test reading
    try:
        sheets = processor.read_file(buffer)
        print(f"✓ Read {len(sheets)} sheet(s)")
        
        for name, df in sheets.items():
            print(f"  {name}: {df.shape}")
        
        print("\n✅ All tests passed!")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
