import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime

from data_loader import get_products_and_interactions
from recommender import HybridRecommender
from image_utils import check_kaggle_cli, download_kaggle_dataset
from ui_components import render_product_card as render_card, render_header

# Page settings
st.set_page_config(
    page_title="🛍️ ShopSmart - Mua sắm thông minh với AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f8f9fa;
        color: #333333;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #fce4ec;
    }
    
    /* Sidebar Navigation Links */
    div[role="radiogroup"] > label > div:first-of-type {
        display: none;
    }
    div[role="radiogroup"] > label {
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 5px;
        transition: background-color 0.2s;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #fff0f6;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #fce4ec;
        color: #e91e63;
        font-weight: 600;
    }
    div[role="radiogroup"] > label p {
        font-size: 1.05em;
        margin: 0;
    }

    /* Metric Cards */
    .dash-metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        display: flex;
        align-items: center;
        border: 1px solid #fce4ec;
        box-shadow: 0 4px 10px rgba(233,30,99,0.05);
        margin-bottom: 20px;
    }
    .dash-icon {
        font-size: 2.2em;
        background: #fdf2f8;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        margin-right: 15px;
        color: #e91e63;
    }
    .dash-info h3 {
        margin: 0;
        font-size: 1.6em;
        color: #333;
        font-weight: 800;
        line-height: 1.2;
    }
    .dash-info p {
        margin: 0;
        font-size: 0.9em;
        color: #666;
        font-weight: 600;
    }
    .dash-info span {
        font-size: 0.8em;
        color: #e91e63;
    }

    /* Header Banner */
    .header-banner { 
        background: linear-gradient(135deg, #ff4081 0%, #f48fb1 100%);
        padding: 35px 20px;
        border-radius: 16px;
        color: white;
        text-align: left;
        box-shadow: 0 10px 25px -5px rgba(233, 30, 99, 0.3);
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
        padding-left: 40px;
    }
    .header-title { 
        font-size: 2.8em; 
        font-weight: 800; 
        margin: 5px 0; 
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .header-subtitle { 
        font-size: 1.05em; 
        opacity: 0.9; 
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Shopee Style Product Grid */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
        gap: 10px;
        margin: 20px 0;
    }
    
    .product-card {
        background: #ffffff;
        border-radius: 2px;
        overflow: hidden;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,.1);
        transition: transform 0.1s cubic-bezier(0.4, 0, 0.6, 1), box-shadow 0.1s cubic-bezier(0.4, 0, 0.6, 1), border 0.1s;
        border: 1px solid transparent;
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        cursor: pointer;
    }
    
    .product-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 1px 20px 0 rgba(0,0,0,.05);
        border: 1px solid #ee4d2d;
        z-index: 1;
    }
    
    .product-image {
        width: 100%;
        padding-top: 100%; /* 1:1 Aspect Ratio */
        position: relative;
        background: #fafafa;
        overflow: hidden;
    }
    
    .product-image img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .discount-badge {
        position: absolute;
        top: 0;
        right: 0;
        width: 36px;
        height: 40px;
        background-color: rgba(255,212,36,.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 2;
    }
    
    .discount-badge::after {
        content: "";
        width: 0;
        height: 0;
        left: 0;
        bottom: -4px;
        position: absolute;
        border-color: transparent rgba(255,212,36,.9);
        border-style: solid;
        border-width: 0 18px 4px;
    }
    
    .discount-percent {
        color: #ee4d2d;
        font-weight: 700;
        font-size: 12px;
        line-height: 1;
        margin-top: 2px;
    }
    
    .discount-label {
        color: #ffffff;
        font-weight: 600;
        font-size: 10px;
        text-transform: uppercase;
        line-height: 1;
        margin-top: 2px;
    }

    .mall-badge {
        position: absolute;
        top: 4px;
        left: -4px;
        background-color: #d0011b;
        color: white;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 2px;
        z-index: 2;
        box-shadow: 0 1px 1px rgba(0,0,0,0.2);
    }
    .mall-badge::before {
        content: "";
        position: absolute;
        left: 0;
        bottom: -3px;
        border-top: 3px solid #900000;
        border-left: 3px solid transparent;
    }
    
    .product-info {
        padding: 8px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        background-color: #fff;
    }
    
    .product-name {
        font-size: 12px;
        color: rgba(0,0,0,.87);
        line-height: 14px;
        height: 28px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        margin-bottom: 4px;
        word-wrap: break-word;
    }
    
    .product-promo-tags {
        display: flex;
        margin-bottom: 6px;
        min-height: 16px;
    }
    
    .promo-tag {
        font-size: 10px;
        color: #ee4d2d;
        border: 1px solid #ee4d2d;
        padding: 1px 4px;
        border-radius: 2px;
        background: transparent;
    }
    
    .product-price-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: auto;
        margin-bottom: 4px;
    }
    
    .product-price {
        color: #ee4d2d;
        font-size: 16px;
        font-weight: 500;
    }
    
    .product-price .currency {
        font-size: 12px;
        margin-right: 1px;
    }
    
    .product-sold {
        font-size: 12px;
        color: rgba(0,0,0,.54);
    }
    
    .product-meta-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 4px;
    }
    
    .product-rating {
        font-size: 8px;
        color: #ffce3d;
        display: flex;
        align-items: center;
        letter-spacing: -1px;
    }
    
    .product-location {
        font-size: 12px;
        color: rgba(0,0,0,.54);
    }
    
    /* Recommendations Explanation */
    .rec-card {
        background: #fdf2f8;
        padding: 20px;
        border-radius: 16px;
        margin: 15px 0;
        border-left: 4px solid #e91e63;
        border-top: 1px solid #fce4ec;
        border-right: 1px solid #fce4ec;
        border-bottom: 1px solid #fce4ec;
        box-shadow: 0 4px 15px rgba(233, 30, 99, 0.05);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4em;
        font-weight: 700;
        color: #333333;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #fce4ec;
        letter-spacing: -0.5px;
    }
    
    /* Footer styles */
    hr {
        border-color: rgba(233,30,99,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Setup session state default variables for settings
    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "kaggle"
        st.session_state['amazon_category'] = "Digital_Music"
        st.session_state['top_n'] = 24
        st.session_state['svd_w'] = 0.55
        st.session_state['item_w'] = 0.30
        st.session_state['content_w'] = 0.15

    data_source = st.session_state['data_source']
    amazon_category = st.session_state['amazon_category']

    # Initialize Recommender Model inside Session State
    if 'recommender' not in st.session_state or st.session_state.get('data_source_current') != (data_source, amazon_category):
        with st.spinner("Đang khởi tạo mô hình gợi ý AI và nạp dữ liệu (chỉ chạy lần đầu)..."):
            products_raw, interactions_raw = get_products_and_interactions(source=data_source, amazon_category=amazon_category)
            model = HybridRecommender(products_raw, interactions_raw)
            model.train_collaborative()
            st.session_state['recommender'] = model
            st.session_state['products'] = products_raw
            st.session_state['interactions'] = interactions_raw
            st.session_state['data_source_current'] = (data_source, amazon_category)
            st.session_state['cart'] = []
            st.session_state['selected_user'] = None

    model = st.session_state['recommender']
    products = st.session_state['products']
    interactions = st.session_state['interactions']

    if products.empty or interactions.empty:
        st.error("❌ Không thể nạp được dữ liệu. Vui lòng kiểm tra lại file dữ liệu hoặc kết nối mạng.")
        return

    # User Simulator
    users = sorted(interactions['user_id'].unique())
    if st.session_state['selected_user'] not in users:
        st.session_state['selected_user'] = users[0]
    
    selected_user = st.session_state['selected_user']
    user_history = interactions[interactions['user_id'] == selected_user]

    # TOP PROFILE BAR
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; gap: 15px; margin-bottom: 20px;">
        <div style="font-size: 1.2em; cursor: pointer;">🔔<sup style="background: #e91e63; color: white; border-radius: 50%; padding: 2px 6px; font-size: 0.6em; margin-left: -5px;">2</sup></div>
        <div style="font-size: 1.2em; cursor: pointer;">🌙</div>
        <div style="display: flex; align-items: center; gap: 10px; background: white; padding: 5px 15px; border-radius: 30px; border: 1px solid #fce4ec; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <div style="background: #fdf2f8; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.2em; color: #e91e63;">👤</div>
            <div style="line-height: 1.2;">
                <div style="font-weight: 700; font-size: 0.85em; color: #333;">{selected_user[:10]}</div>
                <div style="font-size: 0.7em; color: #888;">Đang mô phỏng</div>
            </div>
            <div style="font-size: 0.8em; margin-left: 5px; color: #888;">▼</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.markdown("<h2 style='color:#e91e63; margin-top: -20px; margin-bottom: 30px;'>🛍️ ShopSmart AI</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#888; font-size:0.8em; font-weight:700; letter-spacing: 0.5px;'>MENU CHÍNH</p>", unsafe_allow_html=True)
        menu = st.radio(
            "Navigation",
            ["🏠 Tổng quan", "🔍 Khám phá & Tìm kiếm", "🎯 Gợi ý cá nhân hóa", "📦 Thường mua cùng", "🔥 Sản phẩm hot", "🛒 Giỏ hàng"],
            label_visibility="collapsed",
            key="nav_menu"
        )
        
        st.markdown("<p style='color:#888; font-size:0.8em; font-weight:700; letter-spacing: 0.5px; margin-top: 30px;'>THỐNG KÊ & CÀI ĐẶT</p>", unsafe_allow_html=True)
        admin_menu = st.radio(
            "Admin Navigation",
            ["--- Mặc định ---", "📊 Số liệu hệ thống", "👥 Khách hàng", "⚙️ Cài đặt AI", "📖 Tài liệu hướng dẫn"],
            label_visibility="collapsed",
            index=0,
            key="admin_menu"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #fff0f6; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 10px;">
            <div style="font-size: 2em; margin-bottom: 10px;">🎁</div>
            <h4 style="color: #e91e63; margin: 0 0 10px 0; font-size: 1em;">Trải nghiệm AI cá nhân hóa</h4>
            <p style="font-size: 0.8em; color: #666; margin-bottom: 15px;">AI của chúng tôi đang âm thầm cá nhân hóa trải nghiệm mua sắm của bạn.</p>
        </div>
        """, unsafe_allow_html=True)
        
        def nav_to_admin(menu_name):
            st.session_state['admin_menu'] = menu_name
            
        st.button("Tìm hiểu thêm →", use_container_width=True, type="primary", key="btn_learn_more", on_click=nav_to_admin, args=("📖 Tài liệu hướng dẫn",))

    # Determine Active Route
    active_menu = menu
    if admin_menu != "--- Mặc định ---":
        active_menu = admin_menu

    # Helper function
    def display_product_grid(item_ids, prefix="grid", reasons_list=None):
        num_cols = 4
        num_items = len(item_ids)
        if num_items == 0:
            st.info("Không tìm thấy sản phẩm nào.")
            return

        for i in range(0, num_items, num_cols):
            row_items = item_ids[i:i+num_cols]
            cols = st.columns(num_cols)
            for idx, item_id in enumerate(row_items):
                with cols[idx]:
                    html = render_card(item_id, products)
                    st.markdown(html, unsafe_allow_html=True)
                    
                    if reasons_list and i + idx < len(reasons_list):
                        reasons = reasons_list[i + idx]
                        st.markdown("<div style='margin-top: -10px; margin-bottom: 8px;'>", unsafe_allow_html=True)
                        for r in reasons:
                            st.markdown(f"<span style='font-size:0.7em; background-color:#fce4ec; color:#d81b60; padding:3px 8px; border-radius:10px; margin-right:4px; font-weight:600; display:inline-block; border: 1px solid #f48fb1;'>💡 {r}</span>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    product_row = products[products['item_id'] == item_id]
                    p_name = product_row.iloc[0]['name'] if not product_row.empty else str(item_id)

                    col_btn1, col_btn2 = st.columns([1, 1])
                    with col_btn1:
                        user_likes = st.session_state['interactions']
                        is_liked = not user_likes[(user_likes['user_id'] == selected_user) & (user_likes['item_id'] == item_id)].empty
                        like_label = "💖 Thích" if is_liked else "❤️ Thích"
                        if st.button(like_label, key=f"like_{item_id}_{prefix}_{i}_{idx}", use_container_width=True):
                            model.add_interaction(selected_user, item_id, 5.0)
                            if is_liked:
                                st.session_state['interactions'].loc[
                                    (st.session_state['interactions']['user_id'] == selected_user) & 
                                    (st.session_state['interactions']['item_id'] == item_id), 'rating'
                                ] = 5.0
                            else:
                                new_interaction = pd.DataFrame([{'user_id': selected_user, 'item_id': item_id, 'rating': 5.0}])
                                st.session_state['interactions'] = pd.concat([st.session_state['interactions'], new_interaction], ignore_index=True)
                            st.toast(f"Đã cập nhật sở thích!")
                            time.sleep(0.5)
                            st.rerun()

                    with col_btn2:
                        is_in_cart = item_id in st.session_state['cart']
                        cart_label = "🛒 Đã thêm" if is_in_cart else "🛒+ Giỏ"
                        if st.button(cart_label, key=f"cart_{item_id}_{prefix}_{i}_{idx}", use_container_width=True):
                            if not is_in_cart:
                                st.session_state['cart'].append(item_id)
                                st.toast(f"Đã thêm vào giỏ hàng!")
                                time.sleep(0.5)
                                st.rerun()

    top_n = st.session_state['top_n']
    svd_w = st.session_state['svd_w']
    item_w = st.session_state['item_w']
    content_w = st.session_state['content_w']

    # -------------------------------------------------------------
    # PAGE: TỔNG QUAN (Main Dashboard)
    # -------------------------------------------------------------
    if active_menu == "🏠 Tổng quan":
        # Header Banner
        st.markdown("""
        <div class='header-banner'>
            <div class='header-title'>ShopSmart AI</div>
            <div class='header-subtitle' style='margin-top: 15px; font-size: 1.1em;'>Hệ thống mua sắm thông minh hỗ trợ bởi mô hình gợi ý<br>kết hợp SVD, Item-Item & TF-IDF</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="dash-metric-card">
                <div class="dash-icon">📦</div>
                <div class="dash-info">
                    <p>Số lượng sản phẩm</p>
                    <h3>{len(products):,}</h3>
                    <span>Sản phẩm</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="dash-metric-card">
                <div class="dash-icon" style="color: #3b82f6; background: #eff6ff;">👥</div>
                <div class="dash-info">
                    <p>Số lượng khách hàng</p>
                    <h3 style="color: #1d4ed8;">{len(users):,}</h3>
                    <span style="color: #3b82f6;">Khách hàng</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="dash-metric-card">
                <div class="dash-icon" style="color: #10b981; background: #ecfdf5;">📊</div>
                <div class="dash-info">
                    <p>Giao dịch hệ thống</p>
                    <h3 style="color: #047857;">{len(interactions):,}</h3>
                    <span style="color: #10b981;">Giao dịch</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="dash-metric-card">
                <div class="dash-icon" style="color: #8b5cf6; background: #f5f3ff;">👤</div>
                <div class="dash-info">
                    <p>Đang mô phỏng</p>
                    <h3 style="color: #4c1d95;">{selected_user[:10]}</h3>
                    <span style="color: #8b5cf6;">Tài khoản</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Search Bar Area
        st.markdown("<br>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns([5, 2, 2])
        with col_s1:
            search_input = st.text_input("Tìm kiếm", placeholder="🔍 Nhập từ khóa tìm kiếm sản phẩm...", label_visibility="collapsed", key="search_input_overview")
        with col_s2:
            cats = ["Tất cả danh mục"] + sorted(list(products['category'].dropna().unique()))
            selected_cat_input = st.selectbox("Danh mục", cats, label_visibility="collapsed", key="cat_input_overview")
        with col_s3:
            if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True):
                st.session_state['overview_search_query'] = search_input
                st.session_state['overview_selected_cat'] = selected_cat_input
        
        search_query = st.session_state.get('overview_search_query', '')
        selected_cat = st.session_state.get('overview_selected_cat', 'Tất cả danh mục')

        # Horizontal quick links (Faux Tabs to match mockup)
        st.markdown("<style>div.stButton > button { border-radius: 8px; font-weight: 600; background-color: white; border: 1px solid #fce4ec; color: #555; }</style>", unsafe_allow_html=True)
        
        def nav_to_main(menu_name):
            st.session_state['nav_menu'] = menu_name
            st.session_state['admin_menu'] = "--- Mặc định ---"
            
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.button("🎯 Gợi ý cá nhân hóa", use_container_width=True, key="btn_tab_1", on_click=nav_to_main, args=("🎯 Gợi ý cá nhân hóa",))
        with col_t2:
            st.button("📦 Thường mua cùng", use_container_width=True, key="btn_tab_2", on_click=nav_to_main, args=("📦 Thường mua cùng",))
        with col_t3:
            st.button("🔥 Sản phẩm hot", use_container_width=True, key="btn_tab_3", on_click=nav_to_main, args=("🔥 Sản phẩm hot",))
        with col_t4:
            st.button("🛒 Giỏ hàng của bạn", use_container_width=True, key="btn_tab_4", on_click=nav_to_main, args=("🛒 Giỏ hàng",))

        # Display Top Content
        filtered_products = products.copy()
        if selected_cat != "Tất cả danh mục":
            filtered_products = filtered_products[filtered_products['category'] == selected_cat]
        if search_query:
            filtered_products = filtered_products[filtered_products['name'].str.contains(search_query, case=False, na=False)]

        st.markdown(f"<div style='font-weight: 700; margin-bottom: 15px;'>Sản phẩm tìm thấy ({min(len(filtered_products), top_n)} / {len(filtered_products)})</div>", unsafe_allow_html=True)
        items_to_show = filtered_products['item_id'].head(top_n).tolist()
        display_product_grid(items_to_show, prefix="overview")

    # -------------------------------------------------------------
    # PAGE: KHÁM PHÁ & TÌM KIẾM
    # -------------------------------------------------------------
    elif active_menu == "🔍 Khám phá & Tìm kiếm":
        st.markdown("<div class='section-header'>🔍 KHÁM PHÁ & TÌM KIẾM SẢN PHẨM</div>", unsafe_allow_html=True)
        col_search1, col_search2, col_search3 = st.columns([5, 3, 2])
        with col_search1:
            search_input = st.text_input("Nhập từ khóa tìm kiếm tên sản phẩm...", placeholder="Ví dụ: shoes, phone, bag, ...", key="search_input_explore")
        with col_search2:
            cats = ["Tất cả danh mục"] + sorted(list(products['category'].dropna().unique()))
            selected_cat_input = st.selectbox("Lọc theo danh mục", cats, key="cat_input_explore")
        with col_search3:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔍 Tìm kiếm", type="primary", use_container_width=True, key="btn_explore_search"):
                st.session_state['explore_search_query'] = search_input
                st.session_state['explore_selected_cat'] = selected_cat_input

        search_query = st.session_state.get('explore_search_query', '')
        selected_cat = st.session_state.get('explore_selected_cat', 'Tất cả danh mục')

        filtered_products = products.copy()
        if selected_cat != "Tất cả danh mục":
            filtered_products = filtered_products[filtered_products['category'] == selected_cat]
        if search_query:
            filtered_products = filtered_products[filtered_products['name'].str.contains(search_query, case=False, na=False)]

        st.write(f"Đang hiển thị {min(len(filtered_products), top_n)} / {len(filtered_products)} sản phẩm tìm thấy")
        items_to_show = filtered_products['item_id'].head(top_n).tolist()
        display_product_grid(items_to_show, prefix="explore")

    # -------------------------------------------------------------
    # PAGE: GỢI Ý CÁ NHÂN HÓA
    # -------------------------------------------------------------
    elif active_menu == "🎯 Gợi ý cá nhân hóa":
        st.markdown(f"<div class='section-header'>🎯 GỢI Ý CÁ NHÂN HÓA DÀNH CHO BẠN</div>", unsafe_allow_html=True)
        with st.spinner("Đang tính toán gợi ý cá nhân hóa..."):
            personalized = model.personalized_recommendations(
                selected_user, 
                top_n=top_n, 
                svd_w=svd_w, item_w=item_w, content_w=content_w
            )
            
        if personalized:
            rec_item_ids = [item_id for item_id, score, reasons in personalized]
            reasons_list = [reasons for item_id, score, reasons in personalized]
            display_product_grid(rec_item_ids, prefix="personalized", reasons_list=reasons_list)
        else:
            st.info("Hãy đánh dấu 'Thích' một số sản phẩm để AI có thể gợi ý cho bạn!")

    # -------------------------------------------------------------
    # PAGE: THƯỜNG MUA CÙNG
    # -------------------------------------------------------------
    elif active_menu == "📦 Thường mua cùng":
        st.markdown("<div class='section-header'>📦 SẢN PHẨM THƯỜNG ĐƯỢC MUA CÙNG</div>", unsafe_allow_html=True)
        product_options = [f"{item_id}: {products[products['item_id'] == item_id].iloc[0]['name'][:60]}..." for item_id in sorted(products['item_id'].unique())[:80]]
        selected_option = st.selectbox("Chọn sản phẩm đang xem", product_options)
        selected_item_id = selected_option.split(":")[0].strip()
        
        together = model.frequently_bought_together(selected_item_id, top_n=top_n)
        if together:
            together_item_ids = [item_id for item_id, count in together]
            display_product_grid(together_item_ids, prefix="together")
        else:
            st.info("Chưa có đủ dữ liệu lịch sử mua kèm cho sản phẩm này.")

    # -------------------------------------------------------------
    # PAGE: SẢN PHẨM HOT
    # -------------------------------------------------------------
    elif active_menu == "🔥 Sản phẩm hot":
        st.markdown("<div class='section-header'>🔥 SẢN PHẨM THỊNH HÀNH BÁN CHẠY NHẤT</div>", unsafe_allow_html=True)
        trending = model.trending_products(top_n=top_n)
        if trending:
            trending_item_ids = [item_id for item_id, count in trending]
            display_product_grid(trending_item_ids, prefix="trending")
        else:
            st.info("Chưa có dữ liệu thịnh hành.")

    # -------------------------------------------------------------
    # PAGE: GIỎ HÀNG
    # -------------------------------------------------------------
    elif active_menu == "🛒 Giỏ hàng" or active_menu == "🛒 Giỏ hàng của bạn":
        st.markdown("<div class='section-header'>🛒 GIỎ HÀNG CỦA BẠN</div>", unsafe_allow_html=True)
        cart_items = st.session_state.get('cart', [])
        if cart_items:
            total_price = 0
            st.markdown("""
            <div style='display: flex; padding: 10px 0; border-bottom: 1px solid #fce4ec; color: #888; font-size: 0.9em; font-weight: 600;'>
                <div style='flex: 5;'>Sản Phẩm</div>
                <div style='flex: 2; text-align: right;'>Đơn Giá</div>
                <div style='flex: 1; text-align: right;'>Thao Tác</div>
            </div>
            """, unsafe_allow_html=True)
            for idx, item_id in enumerate(cart_items):
                prod = products[products['item_id'] == item_id]
                if not prod.empty:
                    p_row = prod.iloc[0]
                    name = p_row['name']
                    price = p_row['price']
                    total_price += price
                    image_url = p_row.get('image_url')
                    if pd.isna(image_url) or not image_url:
                        import hashlib
                        seed = hashlib.md5(name.encode()).hexdigest()[:10]
                        image_url = f"https://picsum.photos/100/100?random={seed}"
                    
                    with st.container():
                        st.markdown("<div style='padding: 15px 0;'>", unsafe_allow_html=True)
                        c_img, c_info, c_price, c_action = st.columns([1, 4, 2, 1])
                        with c_img:
                            st.image(image_url, use_container_width=True)
                        with c_info:
                            st.markdown(f"<div style='font-weight: 500; color: #333; line-height: 1.4;'>{name}</div>", unsafe_allow_html=True)
                        with c_price:
                            st.markdown(f"<div style='color: #ee4d2d; font-weight: 600; font-size: 1.1em; text-align: right; padding-top: 10px;'>₫{int(price):,}</div>", unsafe_allow_html=True)
                        with c_action:
                            st.markdown("<div style='text-align: right; padding-top: 5px;'>", unsafe_allow_html=True)
                            if st.button("Xóa", key=f"del_{item_id}_{idx}"):
                                st.session_state['cart'].remove(item_id)
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div><hr style='margin: 0; border-top: 1px dashed #fce4ec;'>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='display: flex; justify-content: space-between; align-items: center; background: white; padding: 20px; margin-top: 20px; border-top: 1px solid #fce4ec; box-shadow: 0 -4px 15px rgba(0,0,0,0.03);'>
                <div style='font-size: 1.1em;'>Tổng thanh toán:</div>
                <div style='font-size: 1.8em; font-weight: 700; color: #ee4d2d;'>₫{int(total_price):,}</div>
            </div>
            <br/>
            """, unsafe_allow_html=True)
            
            col_empty, col_btn = st.columns([3, 1])
            with col_btn:
                if st.button("Mua Hàng", type="primary", use_container_width=True):
                    for item_id in cart_items:
                        model.add_interaction(selected_user, item_id, 4.0)
                        new_interaction = pd.DataFrame([{'user_id': selected_user, 'item_id': item_id, 'rating': 4.0}])
                        st.session_state['interactions'] = pd.concat([st.session_state['interactions'], new_interaction], ignore_index=True)
                    st.session_state['cart'] = []
                    st.success("🎉 Đơn hàng đã được đặt thành công!")
                    time.sleep(1.0)
                    st.rerun()
        else:
            st.info("👉 Giỏ hàng của bạn đang trống.")

    # -------------------------------------------------------------
    # PAGE: CÀI ĐẶT AI & THÔNG SỐ
    # -------------------------------------------------------------
    elif active_menu == "⚙️ Cài đặt AI":
        st.markdown("<div class='section-header'>⚙️ CÀI ĐẶT THUẬT TOÁN AI</div>", unsafe_allow_html=True)
        st.markdown("Tinh chỉnh cách hệ thống AI gợi ý sản phẩm và nguồn dữ liệu đang sử dụng.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ⚙️ Cấu hình dữ liệu")
            new_ds = st.selectbox("Chọn nguồn dữ liệu", ["kaggle", "retail", "amazon"], index=["kaggle", "retail", "amazon"].index(st.session_state['data_source']))
            st.session_state['data_source'] = new_ds
            
            if new_ds == 'amazon':
                new_cat = st.selectbox("Chọn danh mục Amazon", ["Digital_Music", "Electronics"], index=["Digital_Music", "Electronics"].index(st.session_state.get('amazon_category', 'Digital_Music')))
                st.session_state['amazon_category'] = new_cat
            
            st.markdown("### 👤 Mô phỏng người dùng")
            new_user = st.selectbox("Chọn khách hàng mô phỏng", users, index=users.index(st.session_state['selected_user']))
            if new_user != st.session_state['selected_user']:
                st.session_state['selected_user'] = new_user
                st.rerun()

        with c2:
            st.markdown("### 🎛️ Trọng số mô hình lai")
            st.session_state['top_n'] = st.slider("Số lượng gợi ý hiển thị", 8, 48, st.session_state['top_n'])
            
            s_w = st.slider("Cộng tác SVD", 0.0, 1.0, st.session_state['svd_w'])
            i_w = st.slider("Cộng tác Item-Item", 0.0, 1.0, st.session_state['item_w'])
            c_w = st.slider("Nội dung TF-IDF", 0.0, 1.0, st.session_state['content_w'])
            
            w_sum = s_w + i_w + c_w
            if w_sum > 0:
                s_w /= w_sum
                i_w /= w_sum
                c_w /= w_sum
            st.session_state['svd_w'] = s_w
            st.session_state['item_w'] = i_w
            st.session_state['content_w'] = c_w
            st.caption(f"Tỷ lệ thực tế: SVD={s_w:.2f} | Item={i_w:.2f} | TF-IDF={c_w:.2f}")

            if st.button("Lưu cấu hình & Khởi động lại AI", type="primary"):
                if 'recommender' in st.session_state:
                    del st.session_state['recommender']
                st.rerun()

    else:
        st.info(f"Bạn đang ở trang: {active_menu}. Tính năng này đang được phát triển.")

    # Footer
    st.markdown("<hr style='border: 0; height: 1px; background: rgba(233,30,99,0.1); margin: 30px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 15px; font-size: 0.85em; font-weight: 500;'>
        <p>🤖 <b>ShopSmart AI Dashboard</b> — Nền tảng thương mại điện tử tích hợp thuật toán gợi ý lai tiên tiến</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
