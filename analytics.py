import streamlit as st
from streamlit.components.v1 import html

def inject_google_analytics():
    """
    Inject Google Analytics tracking code (GA4) using Streamlit components.
    This is more reliable than using st.markdown for script injection.
    """
    # Google Analytics tracking code
    ga_script = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-P31FMHW9FQ"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-P31FMHW9FQ', { 'debug_mode': true });
    </script>
    
    <!-- Empty div to avoid affecting page layout -->
    <div style="display:none">GA Injected</div>
    """
    
    # Use Streamlit's html component to inject the script
    # This creates an iframe that properly executes the script
    html(ga_script, height=0)

def track_page_view(page_name):
    """
    Track a page view in Google Analytics
    
    Args:
        page_name: Name of the page being viewed
    """
    track_script = f"""
    <script>
        if (typeof gtag !== 'undefined') {{
            gtag('event', 'page_view', {{
                page_title: '{page_name}',
                page_location: window.location.href,
                page_path: '/{page_name.replace(" ", "_").replace("-", "_").lower()}'
            }});
            console.log('Tracked page view: {page_name}');
        }} else {{
            console.log('gtag not defined yet');
        }}
    </script>
    <div style="display:none">Page view tracked: {page_name}</div>
    """
    
    html(track_script, height=0)