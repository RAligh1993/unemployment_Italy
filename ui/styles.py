"""
Custom CSS Styling for Streamlit
Professional theme and components
"""


def get_custom_css() -> str:
    """
    Get complete custom CSS for Streamlit app
    
    Returns:
        CSS string
    """
    
    css = """
    <style>
    /* ========================================
       GLOBAL STYLES
       ======================================== */
    
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-color: #003366;
        --secondary-color: #FF6B35;
        --success-color: #28A745;
        --warning-color: #FFC107;
        --danger-color: #DC3545;
        --info-color: #17A2B8;
        --light-gray: #F8F9FA;
        --dark-gray: #343A40;
        --border-color: #E0E0E0;
        --text-color: #2C3E50;
    }
    
    /* Main app container */
    .stApp {
        background-color: var(--light-gray);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: var(--primary-color);
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* ========================================
       CARDS AND CONTAINERS
       ======================================== */
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    
    .metric-delta {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive {
        color: var(--success-color);
    }
    
    .metric-delta.negative {
        color: var(--danger-color);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    /* ========================================
       BUTTONS
       ======================================== */
    
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, #0055A4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #0055A4 0%, var(--primary-color) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: var(--success-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: #218838;
        transform: translateY(-2px);
    }
    
    /* ========================================
       SIDEBAR
       ======================================== */
    
    .css-1d391kg {
        background-color: white;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] {
        background-color: white;
        border-right: 2px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--primary-color);
    }
    
    /* ========================================
       TABS
       ======================================== */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-color);
        font-weight: 500;
        font-size: 1rem;
        padding: 0 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--light-gray);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, #0055A4 100%);
        color: white;
    }
    
    /* ========================================
       DATAFRAMES AND TABLES
       ======================================== */
    
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe thead tr th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
        padding: 1rem;
        text-align: left;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    .dataframe tbody tr:hover {
        background-color: #e9ecef;
    }
    
    .dataframe tbody tr td {
        padding: 0.75rem 1rem;
    }
    
    /* ========================================
       FILE UPLOADER
       ======================================== */
    
    .stFileUploader {
        background-color: white;
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background-color: #f8f9fa;
    }
    
    /* ========================================
       PROGRESS BARS
       ======================================== */
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }
    
    /* ========================================
       ALERTS AND MESSAGES
       ======================================== */
    
    .stAlert {
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Success alert */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid var(--success-color);
        color: #155724;
    }
    
    /* Info alert */
    .stInfo {
        background-color: #d1ecf1;
        border-left: 4px solid var(--info-color);
        color: #0c5460;
    }
    
    /* Warning alert */
    .stWarning {
        background-color: #fff3cd;
        border-left: 4px solid var(--warning-color);
        color: #856404;
    }
    
    /* Error alert */
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid var(--danger-color);
        color: #721c24;
    }
    
    /* ========================================
       EXPANDERS
       ======================================== */
    
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        padding: 1rem;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--light-gray);
    }
    
    /* ========================================
       SPINNER
       ======================================== */
    
    .stSpinner > div {
        border-top-color: var(--primary-color);
    }
    
    /* ========================================
       PLOTLY CHARTS
       ======================================== */
    
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        background-color: white;
        padding: 1rem;
    }
    
    /* ========================================
       CUSTOM COMPONENTS
       ======================================== */
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.75rem;
        font-weight: 600;
        line-height: 1;
        color: white;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    
    .badge-primary {
        background-color: var(--primary-color);
    }
    
    .badge-success {
        background-color: var(--success-color);
    }
    
    .badge-warning {
        background-color: var(--warning-color);
    }
    
    .badge-danger {
        background-color: var(--danger-color);
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, var(--primary-color) 0%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.875rem;
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
    }
    
    /* ========================================
       RESPONSIVE DESIGN
       ======================================== */
    
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .stButton > button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
    }
    
    </style>
    """
    
    return css


def get_header_html(title: str, subtitle: str = "") -> str:
    """
    Generate HTML for professional header
    
    Args:
        title: Main title
        subtitle: Subtitle (optional)
    
    Returns:
        HTML string
    """
    html = f"""
    <div style="
        background: linear-gradient(135deg, #003366 0%, #0055A4 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        ">{title}</h1>
        {f'<p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    return html
