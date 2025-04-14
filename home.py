import streamlit as st

def home_page():
    """
    Displays the welcome home page with information about the application.
    """
    # Apply custom styling for the text elements
    st.markdown("""
        <style>
        .big-text {
            font-size: 2.6rem !important;
            font-weight: 700 !important;
            color: #1E3A8A;
            line-height: 1.2;
            margin-bottom: 0.5rem;
        }
        .sub-text {
            font-size: 1.4rem !important;
            color: #4B5563;
            margin-bottom: 2rem;
        }
        .centered {
            text-align: center;
        }
        .device-text {
            font-size: 0.85rem;
            color: #4B5563;
            margin-top: 0.5rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        .feature-card {
            background-color: #F9FAFB;
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #2563EB;
            height: 100%;
        }
        .feature-card h3 {
            color: #1E3A8A;
            margin-top: 0;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        .feature-card p {
            color: #4B5563;
            font-size: 1rem;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and subtitle with slightly reduced font sizes
    st.markdown('<h1 class="big-text centered">Transform Your Wearables Data Into Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text centered">Understand patterns and optimize your well-being</p>', unsafe_allow_html=True)
    
    # Show three Garmin watch images with consistent spacing
    img_col1, img_col2, img_col3 = st.columns([1, 1, 1])
    
    with img_col1:
        st.image("https://res.garmin.com/transform/image/upload/b_rgb:FFFFFF,c_pad,dpr_2.0,f_auto,h_600,q_auto,w_600/c_pad,h_600,w_600/v1/Product_Images/en/products/010-02862-13/v/pd-02-xl?pgw=1", 
                 width=100)
        
    with img_col2:
        st.image("https://res.garmin.com/transform/image/upload/b_rgb:FFFFFF,c_pad,dpr_2.0,f_auto,h_600,q_auto,w_600/c_pad,h_600,w_600/v1/Product_Images/en/products/010-02645-00/v/rf-xl-213b7d10-d7aa-4c94-90a1-55fb37a1bb7a?pgw=1", 
                 width=100)
        
    with img_col3:
        st.image("https://res.garmin.com/transform/image/upload/b_rgb:FFFFFF,c_pad,dpr_2.0,f_auto,h_600,q_auto,w_600/c_pad,h_600,w_600/v1/Product_Images/en/products/010-02562-00/v/pd-05-xl-8f6329b0-db1f-4fc3-8320-2934f98fb7bf?pgw=1", 
                 width=100)
    
    # Supported devices text with consistent styling
    st.markdown('<p class="device-text">Currently supported devices: Garmin watches</p>', unsafe_allow_html=True)
    
    # Three key features in columns with equal height cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>Visualize Data</h3>
            <p>See patterns in your physiological data through interactive visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>Annotate Interventions and Events</h3>
            <p>Tag important moments and track how they affect your body's responses over time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>Gain Personalized Insights</h3>
            <p>Discover when interventions positively influence your body's responses.</p>
        </div>
        """, unsafe_allow_html=True)