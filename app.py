"""
Streamlit app for Fashion Shopping Assistant
"""
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64
from io import BytesIO

# Import custom modules
import config
from data_loader import FashionDataLoader
from feature_extractor import FeatureExtractor
from similarity_search import SimilaritySearch

# Page configuration
st.set_page_config(
    page_title="Fashion Shopping Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px;
        text-align: center;
        transition: transform 0.3s;
    }
    .product-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1 {
        color: #1e3d59;
    }
    .stButton button {
        background-color: #ff5757;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load dataset and features"""
    data_loader = FashionDataLoader(use_subset=True, subset_size=2000)
    df = data_loader.load_metadata()

    if df is None:
        return None, None, None, None

    return data_loader, df, None, None

@st.cache_resource
def load_models(_data_loader):
    """Load feature extractor and similarity search"""
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(model_name='resnet50')

    # Check if features already exist
    features_path = config.FEATURES_DIR / f"{config.MODEL_NAME}_features.pkl"

    if features_path.exists():
        # Load pre-computed features
        features, indices = feature_extractor.load_features(features_path)
    else:
        # Extract features
        features, indices = feature_extractor.extract_dataset_features(
            _data_loader, save_path=features_path
        )

    # Initialize similarity search
    similarity_search = SimilaritySearch(features, use_faiss=True)

    return feature_extractor, similarity_search

def display_product(product_info, col):
    """Display a single product card"""
    with col:
        # Display image
        img_path = product_info['image_path']
        if Path(img_path).exists():
            img = Image.open(img_path)
            st.image(img, width="content")
        else:
            st.write("Image not found")

        # Display product info
        st.markdown(f"**{product_info['productDisplayName'][:50]}...**")
        st.caption(f"Category: {product_info['masterCategory']}")
        st.caption(f"Type: {product_info['articleType']}")
        st.caption(f"Color: {product_info['baseColour']}")
        st.caption(f"Gender: {product_info['gender']}")

        # Add to cart button
        if st.button(f"View Details", key=f"view_{product_info['id']}"):
            st.session_state['selected_product'] = product_info

def display_product_grid(products_df, n_cols=4):
    """Display products in a grid layout"""
    n_products = len(products_df)
    n_rows = (n_products + n_cols - 1) // n_cols

    for row_idx in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            product_idx = row_idx * n_cols + col_idx
            if product_idx < n_products:
                product_info = products_df.iloc[product_idx].to_dict()
                display_product(product_info, cols[col_idx])

def main():
    """Main app function"""

    # Title and description
    st.title("🛍️ Fashion Shopping Assistant")
    st.markdown("""
    Welcome to your AI-powered fashion shopping assistant! Browse, search, and discover
    fashion items using advanced image similarity and smart recommendations.
    """)

    # Load data
    with st.spinner("Loading fashion catalog..."):
        data_loader, df, _, _ = load_data()

    if data_loader is None or df is None:
        st.error("""
        Dataset not found! Please ensure the Fashion Product Images dataset is downloaded.
        Run `python download_dataset.py` to download the dataset.
        """)
        return

    # Load models
    with st.spinner("Loading AI models..."):
        feature_extractor, similarity_search = load_models(data_loader)

    # Sidebar filters
    st.sidebar.title("🔍 Search & Filter")

    # Search mode selection
    search_mode = st.sidebar.radio(
        "Search Mode",
        ["Browse Catalog", "Similar Items", "Smart Recommendations", "Upload Image"]
    )

    # Filter options
    st.sidebar.markdown("### Filters")

    selected_gender = st.sidebar.selectbox("Gender", config.GENDER_OPTIONS)
    selected_category = st.sidebar.selectbox("Category", config.MASTER_CATEGORIES)
    selected_color = st.sidebar.selectbox("Color", config.COLOR_OPTIONS)
    selected_season = st.sidebar.selectbox("Season", config.SEASON_OPTIONS)

    # Main content area
    if search_mode == "Browse Catalog":
        st.header("📦 Browse Products")

        # Apply filters
        filtered_df = data_loader.filter_products(
            gender=selected_gender if selected_gender != "All" else None,
            category=selected_category if selected_category != "All" else None,
            color=selected_color if selected_color != "All" else None,
            season=selected_season if selected_season != "All" else None
        )

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Products", len(filtered_df))
        with col2:
            st.metric("Categories", filtered_df['masterCategory'].nunique())
        with col3:
            st.metric("Brands", filtered_df['articleType'].nunique())
        with col4:
            st.metric("Colors", filtered_df['baseColour'].nunique())

        # Pagination
        items_per_page = 12
        n_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
        page = st.slider("Page", 1, max(1, n_pages), 1) - 1

        start_idx = page * items_per_page
        end_idx = min((page + 1) * items_per_page, len(filtered_df))

        # Display products
        if len(filtered_df) > 0:
            display_product_grid(filtered_df.iloc[start_idx:end_idx])
        else:
            st.info("No products found matching your filters.")

    elif search_mode == "Similar Items":
        st.header("🔎 Find Similar Items")

        # Select a product
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select a Product")

            # Random product button
            if st.button("🎲 Random Product"):
                random_idx = np.random.randint(0, len(df))
                st.session_state['query_idx'] = random_idx

            # Product selector
            product_names = df['productDisplayName'].tolist()
            selected_product = st.selectbox(
                "Or choose from catalog",
                product_names,
                index=st.session_state.get('query_idx', 0)
            )

            # Get selected product index
            query_idx = df[df['productDisplayName'] == selected_product].index[0]

            # Display selected product
            query_info = data_loader.get_product_info(query_idx)
            img_path = query_info['image_path']
            if Path(img_path).exists():
                img = Image.open(img_path)
                st.image(img, width="content")

            st.markdown(f"**{query_info['productDisplayName']}**")
            st.caption(f"Category: {query_info['masterCategory']}")
            st.caption(f"Color: {query_info['baseColour']}")

        with col2:
            st.subheader("Similar Items")

            # Get recommendations
            similar_indices, similar_scores = similarity_search.get_recommendations(
                query_idx, k=8, exclude_self=True
            )

            # Display similar items
            similar_df = df.iloc[similar_indices]
            display_product_grid(similar_df, n_cols=4)

    elif search_mode == "Smart Recommendations":
        st.header("🎯 Smart Recommendations")

        # Recommendation type
        rec_type = st.selectbox(
            "Recommendation Type",
            ["Diverse Recommendations", "Complementary Items", "Complete the Outfit"]
        )

        # Select a product
        product_names = df['productDisplayName'].tolist()
        selected_product = st.selectbox(
            "Select a product",
            product_names,
            index=0
        )

        # Get selected product index
        query_idx = df[df['productDisplayName'] == selected_product].index[0]
        query_info = data_loader.get_product_info(query_idx)

        # Display selected product
        col1, col2 = st.columns([1, 3])
        with col1:
            img_path = query_info['image_path']
            if Path(img_path).exists():
                img = Image.open(img_path)
                st.image(img, width="content")
            st.markdown(f"**{query_info['productDisplayName']}**")

        with col2:
            if rec_type == "Diverse Recommendations":
                st.subheader("Diverse Recommendations")
                indices, scores = similarity_search.get_diverse_recommendations(
                    query_idx, k=8, diversity_weight=0.3
                )
                recommended_df = df.iloc[indices]
                display_product_grid(recommended_df, n_cols=4)

            elif rec_type == "Complementary Items":
                st.subheader("Complementary Items")
                indices, scores = similarity_search.find_complementary_items(
                    query_idx, data_loader, k=8
                )
                if len(indices) > 0:
                    recommended_df = df.iloc[indices]
                    display_product_grid(recommended_df, n_cols=4)
                else:
                    st.info("No complementary items found.")

            else:  # Complete the Outfit
                st.subheader("Complete the Outfit")
                outfit_items = similarity_search.find_outfit_items(
                    query_idx, data_loader, k=3
                )

                for category, (indices, scores) in outfit_items.items():
                    st.markdown(f"### {category}")
                    outfit_df = df.iloc[indices]
                    display_product_grid(outfit_df, n_cols=len(indices))

    else:  # Upload Image
        st.header("📸 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Your Image")
                image = Image.open(uploaded_file)
                st.image(image, width="content")

            with col2:
                st.subheader("Similar Items in Catalog")

                # Extract features from uploaded image
                with st.spinner("Analyzing image..."):
                    query_features = feature_extractor.extract_features(image)

                    # Search for similar items
                    distances, indices = similarity_search.search_similar(
                        query_features, k=8
                    )

                    # Display results
                    similar_df = df.iloc[indices[0]]
                    display_product_grid(similar_df, n_cols=4)

    # Dataset Statistics
    with st.expander("📊 Dataset Statistics"):
        stats = data_loader.get_product_stats()
        if stats:
            col1, col2 = st.columns(2)

            with col1:
                # Category distribution
                fig = px.pie(
                    values=list(stats['categories'].values()),
                    names=list(stats['categories'].keys()),
                    title="Product Categories"
                )
                st.plotly_chart(fig, use_full_container_width=True)

            with col2:
                # Gender distribution
                fig = px.bar(
                    x=list(stats['genders'].keys()),
                    y=list(stats['genders'].values()),
                    title="Gender Distribution"
                )
                st.plotly_chart(fig, use_full_container_width=True)

if __name__ == "__main__":
    main()
