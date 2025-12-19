"""
Nowcasting Platform - UI Module
Professional Streamlit components and visualizations
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main UI components
from .styles import get_custom_css, get_header_html
from .components import (
    DashboardComponents,
    WelcomeScreen,
    DataPreviewComponent,
    ModelConfigurationComponent,
    ResultsDashboard
)
from .charts import NowcastingCharts, ChartTheme, InteractiveComponents

# Define what's available for "from ui import *"
__all__ = [
    # Styles
    'get_custom_css',
    'get_header_html',
    
    # Components
    'DashboardComponents',
    'WelcomeScreen',
    'DataPreviewComponent',
    'ModelConfigurationComponent',
    'ResultsDashboard',
    
    # Charts
    'NowcastingCharts',
    'ChartTheme',
    'InteractiveComponents',
]

# UI configuration
UI_CONFIG = {
    'theme': 'istat',
    'primary_color': '#003366',
    'secondary_color': '#FF6B35',
    'default_chart_height': 500,
    'default_chart_width': None,  # Auto
}

def get_ui_config():
    """Get UI configuration"""
    return UI_CONFIG.copy()

def print_info():
    """Print UI module information"""
    info = f"""
    ╔═══════════════════════════════════════════════════════╗
    ║       NOWCASTING PLATFORM - UI MODULE                 ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Version:  {__version__:<40}      ║
    ║  Author:   {__author__:<40}      ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Components:                                          ║
    ║    ✓ Custom CSS Styling                              ║
    ║    ✓ Dashboard Components                            ║
    ║    ✓ Welcome & Preview Screens                       ║
    ║    ✓ Model Configuration UI                          ║
    ║    ✓ Results Dashboard                               ║
    ║    ✓ Professional Charts (Plotly)                    ║
    ║    ✓ Interactive Components                          ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(info)

# Chart theme singleton
_chart_theme = None

def get_chart_theme():
    """Get singleton chart theme instance"""
    global _chart_theme
    if _chart_theme is None:
        _chart_theme = ChartTheme()
    return _chart_theme

# Streamlit page config helper
def configure_streamlit_page(
    title: str = "Nowcasting Platform",
    icon: str = "📈",
    layout: str = "wide"
):
    """
    Configure Streamlit page with consistent settings
    
    Args:
        title: Page title
        icon: Page icon (emoji)
        layout: Layout ('wide' or 'centered')
    
    Example:
        >>> from ui import configure_streamlit_page
        >>> configure_streamlit_page()
    """
    import streamlit as st
    
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/nowcasting-platform',
            'Report a bug': 'https://github.com/yourusername/nowcasting-platform/issues',
            'About': f"""
            # Nowcasting Platform {__version__}
            
            Professional forecasting tool for economists and data scientists.
            """
        }
    )
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

# Module-level initialization
def _initialize():
    """Initialize UI module"""
    pass

_initialize()
