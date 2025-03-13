# Instructions for running the web app locally

1. Change directories to the **`wearables_web_app`** subfolder in this repo
2. Install streamlit using `pip install streamlit`
    + Check the version: `streamlit --version`
    + If Streamlit is not recognized, it might not be added to your system's PATH variable. You can try adding it manually:
        * On Windows, you can add the path to the Streamlit executable (e.g., `C:\Users\YourUsername\AppData\Local\Programs\Python\PythonXX\Scripts`) to the PATH variable in System Properties.
        * On macOS and Linux, you can add the path to your ~/.bash_profile or ~/.bashrc file: `export PATH="$PATH:/path/to/streamlit"`
3. Install the vega datasets package: `pip install vega_datasets`
4. In the `wearables_web_app` folder, type the following command in the terminal to run the web app: `streamlit run main.py`
5. If you get a Google Cloud related error when trying to run the webapp, try the following commands:
    + `pip install google-cloud`
    + `pip install google-cloud-vision`
    + `pip install --upgrade google-cloud google-api-core google-auth`

# Description of each subfolder in this repo

### wearables_web_app

This is the finalized web app. It is deployed publically/live at: https://wearablesdatainsight.streamlit.app/
