import streamlit as st

def inject_google_analytics():
    ga_script = """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-P31FMHW9FQ"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());

      gtag('config', 'G-P31FMHW9FQ');
    </script>
    """

    # Inject Google Analytics script into the app
    st.markdown(ga_script, unsafe_allow_html=True)