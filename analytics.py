import streamlit as st
from streamlit.components.v1 import html

def inject_google_analytics():
    """
    Minimal GA4 injection for Streamlit.
    Just paste the Google snippet here and call it once at app startup.
    """
    ga_code = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-P31FMHW9FQ"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-P31FMHW9FQ');
    </script>
    """
    # Injecting the script via an invisible HTML component
    html(ga_code, height=0)

def track_page_view(page_name):
    """
    Minimal page_view event for GA4.
    """
    # JavaScript snippet to call gtag('event', 'page_view', ...)
    track_script = f"""
    <script>
      if (typeof gtag !== 'undefined') {{
        gtag('event', 'page_view', {{
          page_title: '{page_name}',
          page_location: window.location.href,
          page_path: '/{page_name.lower().replace(" ", "-")}'
        }});
        console.log('Tracked page view: {page_name}');
      }} else {{
        console.log('gtag not defined yet');
      }}
    </script>
    """
    # Inject the tracking script via an invisible HTML component
    html(track_script, height=0)
