import streamlit as st
from streamlit.components.v1 import html

def inject_google_analytics():
    """
    Inject Google Analytics (GA4) tracking code into the Streamlit app.
    """
    ga_script = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-P31FMHW9FQ"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-P31FMHW9FQ');
    </script>
    <!-- Adding a visible element ensures the iframe is properly rendered -->
    <div style="display:none">GA Initialized</div>
    """
    
    # Use html component for reliable script execution
    html(ga_script, height=0)

def track_page_view(page_name):
    """
    Track a page view in Google Analytics.
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
    <!-- Adding a visible element ensures the iframe is properly rendered -->
    <div style="display:none">Page view tracked: {page_name}</div>
    """
    
    # Use html component for reliable script execution
    html(track_script, height=0)
