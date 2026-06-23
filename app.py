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
from ui_components import render_product_card as render_card, render_header, render_cart_item

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
    div[role="radiogroup"] {
        gap: 8px;
    }
    div[role="radiogroup"] > label {
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 2px;
        transition: all 0.2s ease;
        position: relative;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #f8f9fa;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #fff0f6;
    }
    div[role="radiogroup"] > label[data-checked="true"] p {
        color: #e91e63 !important;
        font-weight: 700;
    }
    div[role="radiogroup"] > label p {
        font-size: 1.05em;
        margin: 0;
        color: #555;
        font-weight: 500;
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
    
    /* Cart Item Horizontal */
    .cart-item {
        display: flex;
        background: #fff;
        border: 1px solid #fce4ec;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 2px;
        align-items: center;
        box-shadow: 0 2px 5px rgba(233,30,99,0.05);
    }
    .cart-item-image {
        width: 70px;
        height: 70px;
        object-fit: cover;
        border-radius: 4px;
        margin-right: 15px;
        border: 1px solid #eee;
    }
    .cart-item-info {
        flex-grow: 1;
    }
    .cart-item-name {
        font-size: 14px;
        font-weight: 600;
        color: #333;
        margin-bottom: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .cart-item-category {
        font-size: 12px;
        color: #888;
    }
    .cart-item-price {
        font-size: 16px;
        color: #ee4d2d;
        font-weight: 600;
        width: 120px;
        text-align: right;
        margin-right: 10px;
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

USERS_FILE = "users.json"
CUSTOM_INTERACTIONS_FILE = "custom_interactions.csv"

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users_dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users_dict, f)

def save_custom_interaction(user_id, item_id, rating):
    df = pd.DataFrame([{'user_id': user_id, 'item_id': item_id, 'rating': float(rating)}])
    if not os.path.exists(CUSTOM_INTERACTIONS_FILE):
        df.to_csv(CUSTOM_INTERACTIONS_FILE, index=False)
    else:
        df.to_csv(CUSTOM_INTERACTIONS_FILE, mode='a', header=False, index=False)

def main():
    # Setup session state default variables for settings
    if 'logged_in_user' not in st.session_state:
        st.session_state['logged_in_user'] = None
    if 'show_login' not in st.session_state:
        st.session_state['show_login'] = False

    if 'data_source' not in st.session_state:
        st.session_state['data_source'] = "electronics"
    if 'top_n' not in st.session_state:
        st.session_state['top_n'] = 100
        st.session_state['amazon_category'] = "Digital_Music"
        st.session_state['svd_w'] = 0.55
        st.session_state['item_w'] = 0.30
        st.session_state['content_w'] = 0.15

    data_source = st.session_state['data_source']
    amazon_category = st.session_state['amazon_category']

    # Initialize Recommender Model inside Session State
    if 'recommender' not in st.session_state or st.session_state.get('data_source_current') != (data_source, amazon_category):
        with st.spinner("Đang khởi tạo mô hình gợi ý AI và nạp dữ liệu (chỉ chạy lần đầu)..."):
            products_raw, interactions_raw = get_products_and_interactions(source=data_source, amazon_category=amazon_category)
            
            if os.path.exists(CUSTOM_INTERACTIONS_FILE):
                try:
                    custom_df = pd.read_csv(CUSTOM_INTERACTIONS_FILE)
                    interactions_raw = pd.concat([interactions_raw, custom_df], ignore_index=True)
                except:
                    pass

            model = HybridRecommender(products_raw, interactions_raw)
            model.train_collaborative()
            st.session_state['recommender'] = model
            st.session_state['products'] = products_raw
            st.session_state['interactions'] = interactions_raw
            st.session_state['data_source_current'] = (data_source, amazon_category)
            st.session_state['cart'] = {}

    if 'cart' in st.session_state and isinstance(st.session_state['cart'], list):
        st.session_state['cart'] = {item: 1 for item in st.session_state['cart']}

    model = st.session_state['recommender']
    products = st.session_state['products']
    interactions = st.session_state['interactions']

    if products.empty or interactions.empty:
        st.error("❌ Không thể nạp được dữ liệu. Vui lòng kiểm tra lại file dữ liệu hoặc kết nối mạng.")
        return

    # Set selected_user to the currently logged in user
    selected_user = st.session_state['logged_in_user']

    # TOP PROFILE BAR
    if st.session_state['logged_in_user']:
        col_space, col_cart, col_user_txt, col_logout = st.columns([8, 1.8, 1.2, 1.2])
        with col_cart:
            cart_count = sum(st.session_state.get('cart', {}).values())
            if st.button(f"🛒 Giỏ hàng ({cart_count})", use_container_width=True):
                st.session_state['nav_menu'] = "🛒 Giỏ hàng"
                st.rerun()
        with col_user_txt:
            st.markdown(f"<div style='margin-top: 6px; text-align: center; font-weight: 800; color: #e91e63; font-size: 1.2em; letter-spacing: 0.5px;'>👤 {st.session_state['logged_in_user']}</div>", unsafe_allow_html=True)
        with col_logout:
            if st.button("Đăng xuất", use_container_width=True):
                st.session_state['logged_in_user'] = None
                st.rerun()
    else:
        col_space, col_cart, col_login = st.columns([8, 1.8, 2.4])
        with col_cart:
            cart_count = sum(st.session_state.get('cart', {}).values())
            if st.button(f"🛒 Giỏ hàng ({cart_count})", use_container_width=True):
                st.session_state['nav_menu'] = "🛒 Giỏ hàng"
                st.rerun()
        with col_login:
            if st.button("🔑 Đăng nhập / Đăng ký", use_container_width=True, type="primary"):
                st.session_state['show_login'] = not st.session_state['show_login']
                st.rerun()

    # Login / Register UI
    if st.session_state['show_login']:
        st.markdown("<div class='section-header'>🔑 Hệ thống Tài khoản</div>", unsafe_allow_html=True)
        tab_login, tab_reg = st.tabs(["🔐 Đăng nhập", "📝 Đăng ký mới"])
        
        with tab_login:
            username = st.text_input("Tên đăng nhập", key="login_user")
            password = st.text_input("Mật khẩu", type="password", key="login_pass")
            if st.button("Đăng nhập", type="primary", key="btn_login"):
                users_db = load_users()
                if username in users_db and users_db[username] == password:
                    st.session_state['logged_in_user'] = username
                    st.session_state['show_login'] = False
                    st.success("Đăng nhập thành công!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Sai tên đăng nhập hoặc mật khẩu!")
                    
        with tab_reg:
            new_user = st.text_input("Tên đăng nhập mới", key="reg_user")
            new_pass = st.text_input("Mật khẩu", type="password", key="reg_pass")
            if st.button("Đăng ký", type="primary", key="btn_register"):
                users_db = load_users()
                if new_user in users_db:
                    st.error("Tên đăng nhập đã tồn tại!")
                elif len(new_user) < 3:
                    st.error("Tên đăng nhập phải có ít nhất 3 ký tự!")
                else:
                    users_db[new_user] = new_pass
                    save_users(users_db)
                    st.session_state['logged_in_user'] = new_user
                    st.session_state['show_login'] = False
                    st.success("Đăng ký thành công! Đang tự động đăng nhập...")
                    time.sleep(0.5)
                    st.rerun()
        st.markdown("<hr style='border: 0; height: 1px; background: rgba(233,30,99,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
            
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # SIDEBAR NAVIGATION
    with st.sidebar:
        st.markdown("<h2 style='color:#e91e63; margin-top: -20px; margin-bottom: 30px;'>🛍️ Smart Electronics</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='color:#888; font-size:0.8em; font-weight:700; letter-spacing: 0.5px;'>MENU CHÍNH</p>", unsafe_allow_html=True)
        menu = st.radio(
            "Navigation",
            ["🏠 Trang chủ", "🔍 Tìm kiếm", "🛒 Giỏ hàng", "💖 Yêu thích", "🎯 Gợi ý cho bạn", "🔥 Sản phẩm hot", "⚙️ Cài đặt AI"],
            label_visibility="collapsed",
            key="nav_menu"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: linear-gradient(180deg, #fff0f6 0%, #ffffff 100%); padding: 25px 20px; border-radius: 16px; text-align: center; margin-bottom: 20px; border: 1px solid #fce4ec; box-shadow: 0 4px 12px rgba(233,30,99,0.05);">
            <div style="font-size: 3em; margin-bottom: 10px;">🎁</div>
            <h4 style="color: #e91e63; margin: 0 0 10px 0; font-size: 1.05em; font-weight: 800;">Trải nghiệm AI cá nhân hóa</h4>
            <p style="font-size: 0.85em; color: #666; margin-bottom: 20px; line-height: 1.4;">AI của chúng tôi đang âm thầm cá nhân<br>hóa trải nghiệm mua sắm của bạn.</p>
            <div style="background-color: #f43f5e; color: white; padding: 10px 15px; border-radius: 8px; font-weight: 600; font-size: 0.9em; cursor: pointer; transition: all 0.2s;">Khám phá ngay</div>
        </div>
        <div style="text-align: center; color: #888; font-size: 0.75em; margin-top: 30px;">
            <p>© 2024 Smart Electronics</p>
            <p style="letter-spacing: 5px; font-size: 1.2em; color: #aaa;">ⓕ 📸 ✉</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Determine Active Route
    active_menu = menu

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
                    
                    product_row = products[products['item_id'] == item_id]
                    p_name = product_row.iloc[0]['name'] if not product_row.empty else str(item_id)

                    col_btn1, col_btn2 = st.columns([1, 1])
                    with col_btn1:
                        user_likes = st.session_state['interactions']
                        is_liked = False
                        if selected_user:
                            is_liked = not user_likes[(user_likes['user_id'] == selected_user) & (user_likes['item_id'] == item_id)].empty
                        
                        like_label = "💖 Thích" if is_liked else "❤️ Thích"
                        if st.button(like_label, key=f"like_{item_id}_{prefix}_{i}_{idx}", use_container_width=True):
                            if not selected_user:
                                st.warning("Vui lòng Đăng nhập để Thích sản phẩm!")
                            else:
                                model.add_interaction(selected_user, item_id, 5.0)
                                if is_liked:
                                    st.session_state['interactions'].loc[
                                        (st.session_state['interactions']['user_id'] == selected_user) & 
                                        (st.session_state['interactions']['item_id'] == item_id), 'rating'
                                    ] = 5.0
                                else:
                                    new_interaction = pd.DataFrame([{'user_id': selected_user, 'item_id': item_id, 'rating': 5.0}])
                                    st.session_state['interactions'] = pd.concat([st.session_state['interactions'], new_interaction], ignore_index=True)
                                
                                save_custom_interaction(selected_user, item_id, 5.0)
                                st.toast(f"Đã cập nhật sở thích cho {selected_user}!")
                                time.sleep(0.5)
                                st.rerun()

                    with col_btn2:
                        cart_label = "🛒+ Giỏ"
                        if st.button(cart_label, key=f"cart_{item_id}_{prefix}_{i}_{idx}", use_container_width=True):
                            if item_id not in st.session_state['cart']:
                                st.session_state['cart'][item_id] = 1
                            else:
                                st.session_state['cart'][item_id] += 1
                            st.toast(f"Đã thêm vào giỏ hàng!")
                            time.sleep(0.5)
                            st.rerun()

    top_n = st.session_state['top_n']
    svd_w = st.session_state['svd_w']
    item_w = st.session_state['item_w']
    content_w = st.session_state['content_w']

    # --- GLOBAL HEADER & NAVIGATION ---
    if active_menu not in ["🛒 Giỏ hàng", "⚙️ Cài đặt AI"]:
        st.markdown("""
        <style>
        .header-banner {
            background: linear-gradient(135deg, #ff4081 0%, #9c27b0 100%) !important;
            padding: 45px 50px !important;
            border-radius: 20px !important;
            box-shadow: 0 15px 30px rgba(156, 39, 176, 0.2) !important;
            overflow: hidden;
            position: relative;
            margin-bottom: 30px;
        }
        
        /* Search Box - 6th child of main block */
        div[data-testid="stMain"] .element-container:nth-child(6) div[data-testid="stTextInput"] input {
            background-color: #f8f9fa !important;
            border: 1px solid #f8f9fa !important;
            border-radius: 12px !important;
            padding: 14px 20px !important;
            font-size: 1.05em !important;
            box-shadow: none !important;
        }
        div[data-testid="stMain"] .element-container:nth-child(6) div[data-testid="stTextInput"] input:focus {
            border: 1px solid #ff4081 !important;
            background-color: #ffffff !important;
        }
        div[data-testid="stMain"] .element-container:nth-child(6) div[data-testid="stButton"] button {
            background-color: #f43f5e !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            height: 52px !important;
            transition: all 0.2s !important;
        }
        div[data-testid="stMain"] .element-container:nth-child(6) div[data-testid="stButton"] button:hover {
            background-color: #e11d48 !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(244, 63, 94, 0.3) !important;
        }
        
        /* Quick Links - 7th child of main block */
        div[data-testid="stMain"] .element-container:nth-child(7) div[data-testid="stButton"] button {
            background-color: #ffffff !important;
            border: 1px solid #eaeaea !important;
            border-radius: 20px !important;
            padding: 8px 15px !important;
            font-size: 0.9em !important;
            font-weight: 600 !important;
            color: #555 !important;
            transition: all 0.2s !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
        }
        div[data-testid="stMain"] .element-container:nth-child(7) div[data-testid="stButton"] button:hover {
            border-color: #e91e63 !important;
            color: #e91e63 !important;
            transform: translateY(-1px);
        }
        div[data-testid="stMain"] .element-container:nth-child(7) div[data-testid="stButton"] button p {
            font-size: 0.95em; margin: 0; padding: 0; white-space: nowrap;
        }
        
        /* Main Tabs - 8th child of main block */
        div[data-testid="stMain"] .element-container:nth-child(8) div[data-testid="stButton"] button {
            height: 100px !important;
            background: white !important;
            border: 1px solid #eaeaea !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03) !important;
            color: #333 !important;
            justify-content: flex-start !important;
            padding: 0 25px !important;
            position: relative;
            transition: all 0.2s !important;
        }
        div[data-testid="stMain"] .element-container:nth-child(8) div[data-testid="stButton"] button:hover {
            border-color: #e91e63 !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 20px rgba(233,30,99,0.08) !important;
        }
        div[data-testid="stMain"] .element-container:nth-child(8) div[data-testid="stButton"] button p {
            font-size: 1.15em;
            font-weight: 800;
        }
        div[data-testid="stMain"] .element-container:nth-child(8) div[data-testid="stButton"] button::after {
            content: "→";
            position: absolute;
            right: 25px;
            top: 50%;
            transform: translateY(-50%);
            color: #3b82f6;
            font-size: 1.6em;
            font-weight: 500;
        }
        div[data-testid="stMain"] .element-container:nth-child(8) div[data-testid="column"]:nth-child(1) div[data-testid="stButton"] button p::after {
            content: "\\A Khám phá sản phẩm phù hợp với sở thích";
            white-space: pre;
            font-size: 0.7em;
            color: #888;
            font-weight: 500;
            display: block;
            margin-top: 5px;
        }
        div[data-testid="stMain"] .element-container:nth-child(8) div[data-testid="column"]:nth-child(2) div[data-testid="stButton"] button p::after {
            content: "\\A Những sản phẩm được yêu thích nhất hiện nay";
            white-space: pre;
            font-size: 0.7em;
            color: #888;
            font-weight: 500;
            display: block;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Single line HTML to prevent Markdown code block parsing
        banner_html = "<div class='header-banner'><div style='position: absolute; right: 20%; top: -10%; font-size: 8em; opacity: 0.8; transform: rotate(15deg); filter: drop-shadow(0 10px 10px rgba(0,0,0,0.1));'>🛍️</div><div style='position: absolute; right: 5%; top: 20%; font-size: 5em; opacity: 0.6; transform: rotate(-15deg); filter: drop-shadow(0 10px 10px rgba(0,0,0,0.1));'>🎧</div><div style='position: absolute; right: 35%; bottom: 10%; font-size: 4em; opacity: 0.7; transform: rotate(10deg); filter: drop-shadow(0 10px 10px rgba(0,0,0,0.1));'>🔊</div><div style='position: absolute; left: 45%; top: 15%; font-size: 2em; opacity: 0.9; color: white;'>✨</div><div style='position: absolute; left: 52%; top: 25%; font-size: 1.5em; opacity: 0.9; color: white;'>✨</div><div class='header-title' style='font-size: 3.2em; font-weight: 800; margin-bottom: 15px; position: relative; z-index: 2;'>ShopSmart AI</div><div class='header-subtitle' style='font-size: 1.15em; line-height: 1.5; font-weight: 500; opacity: 0.95; position: relative; z-index: 2;'>Hệ thống mua sắm thông minh hỗ trợ bởi mô hình gợi ý<br>kết hợp SVD, Item-Item & TF-IDF</div></div>"
        st.markdown(banner_html, unsafe_allow_html=True)

        def handle_search_click():
            search_query = st.session_state.get('search_input_overview', '')
            st.session_state['overview_search_query'] = search_query
            st.session_state['overview_selected_cat'] = "Tất cả danh mục"
            st.session_state['nav_menu'] = "🏠 Trang chủ"

        col_s1, col_s3 = st.columns([7, 2])
        with col_s1:
            st.text_input("Tìm kiếm", placeholder="Nhập từ khóa tìm kiếm sản phẩm...", label_visibility="collapsed", key="search_input_overview")
        with col_s3:
            st.button("🔍 Tìm kiếm", type="primary", use_container_width=True, on_click=handle_search_click)

        # Quick search links below the search bar
        quick_keywords = [("📱", "Điện thoại"), ("🎧", "Tai nghe"), ("⌨️", "Bàn phím"), ("🖱️", "Chuột"), ("🔊", "Loa"), ("🔌", "Cáp sạc"), ("🔋", "Sạc dự phòng"), ("📱", "Ốp lưng")]
        cols_qk = st.columns([1.2, 1.1, 1.1, 0.9, 0.8, 1.0, 1.5, 1.0])
        def handle_quick_keyword_click(kw):
            st.session_state['overview_search_query'] = kw
            st.session_state['search_input_overview'] = kw
            st.session_state['overview_selected_cat'] = "Tất cả danh mục"
            st.session_state['nav_menu'] = "🏠 Trang chủ"

        for i, (icon, kw) in enumerate(quick_keywords):
            with cols_qk[i]:
                st.button(f"{icon} {kw}", key=f"qk_{i}", use_container_width=True, on_click=handle_quick_keyword_click, args=(kw,))
        
        def nav_to_main(menu_name):
            st.session_state['nav_menu'] = menu_name
            st.session_state['admin_menu'] = "--- Mặc định ---"
            
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.button("🎯 Gợi ý cho bạn", use_container_width=True, key="btn_tab_1", on_click=nav_to_main, args=("🎯 Gợi ý cho bạn",))
        with col_t2:
            st.button("🔥 Sản phẩm hot", use_container_width=True, key="btn_tab_3", on_click=nav_to_main, args=("🔥 Sản phẩm hot",))

        st.markdown("<hr style='border: 0; height: 1px; background: rgba(233,30,99,0.1); margin: 30px 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # PAGE: TỔNG QUAN (Main Dashboard)
    # -------------------------------------------------------------
    
    def get_search_regex(query):
        quick_regex_map = {
            "Điện thoại": "Phone|Mobile|Samsung|iPhone|Redmi|OnePlus|Realme|Poco|Oppo|Vivo|Xiaomi|Smartphone",
            "Tai nghe": "Ear|Headphone|boAt|Buds|AirPods|Headset",
            "Bàn phím": "Keyboard",
            "Chuột": "Mouse",
            "Loa": "Speaker|Soundbar",
            "Cáp sạc": "Cable|Adapter|Charger|Lightning",
            "Sạc dự phòng": "Power Bank",
            "Ốp lưng": "Case|Cover"
        }
        for kw, rgx in quick_regex_map.items():
            if kw.lower() == str(query).lower():
                return rgx
        return query

    if active_menu == "🏠 Trang chủ":
        search_query = st.session_state.get('overview_search_query', '')
        selected_cat = st.session_state.get('overview_selected_cat', 'Tất cả danh mục')

        filtered_products = products.copy()
        if selected_cat != "Tất cả danh mục":
            filtered_products = filtered_products[filtered_products['category'] == selected_cat]
        if search_query:
            regex_query = get_search_regex(search_query)
            filtered_products = filtered_products[filtered_products['name'].str.contains(regex_query, case=False, na=False)]

        items_to_show = filtered_products['item_id'].head(top_n).tolist()
        display_product_grid(items_to_show, prefix="overview")

    # -------------------------------------------------------------
    # PAGE: KHÁM PHÁ & TÌM KIẾM
    # -------------------------------------------------------------
    elif active_menu == "🔍 Tìm kiếm":
        st.markdown("<div class='section-header'>🔍 KHÁM PHÁ & TÌM KIẾM SẢN PHẨM</div>", unsafe_allow_html=True)

        search_query = st.session_state.get('overview_search_query', '')
        selected_cat = st.session_state.get('overview_selected_cat', 'Tất cả danh mục')

        filtered_products = products.copy()
        if selected_cat != "Tất cả danh mục":
            filtered_products = filtered_products[filtered_products['category'] == selected_cat]
        if search_query:
            regex_query = get_search_regex(search_query)
            filtered_products = filtered_products[filtered_products['name'].str.contains(regex_query, case=False, na=False)]

        st.write(f"Đang hiển thị {min(len(filtered_products), top_n)} / {len(filtered_products)} sản phẩm tìm thấy")
        items_to_show = filtered_products['item_id'].head(top_n).tolist()
        display_product_grid(items_to_show, prefix="explore")

    # -------------------------------------------------------------
    # PAGE: GIỎ HÀNG
    # -------------------------------------------------------------
    elif active_menu == "🛒 Giỏ hàng":
        st.markdown("<div class='section-header'>🛒 GIỎ HÀNG CỦA BẠN</div>", unsafe_allow_html=True)
        cart_items = st.session_state.get('cart', {})
        
        if not cart_items:
            st.info("Giỏ hàng của bạn đang trống. Hãy thêm sản phẩm vào giỏ hàng!")
        else:
            total_items_count = sum(cart_items.values())
            st.markdown(f"**Bạn có {total_items_count} sản phẩm trong giỏ hàng:**")
            
            # Header row like Shopee
            col_h1, col_h2, col_h3 = st.columns([6, 2.5, 1.5])
            with col_h1:
                st.markdown("""
                <div style="display: flex; padding: 10px 15px; background: #fff; border-radius: 8px; margin-bottom: 10px; font-weight: 600; font-size: 13px; color: #888; border: 1px solid #eee;">
                    <div style="flex-grow: 1; padding-left: 85px;">Sản Phẩm</div>
                    <div style="width: 120px; text-align: right; margin-right: 10px;">Đơn Giá</div>
                </div>
                """, unsafe_allow_html=True)
            with col_h2:
                st.markdown("""
                <div style="padding: 10px 0; background: #fff; border-radius: 8px; margin-bottom: 10px; font-weight: 600; font-size: 13px; color: #888; border: 1px solid #eee; text-align: center;">
                    Số Lượng
                </div>
                """, unsafe_allow_html=True)
            with col_h3:
                st.markdown("""
                <div style="padding: 10px 0; background: #fff; border-radius: 8px; margin-bottom: 10px; font-weight: 600; font-size: 13px; color: #888; border: 1px solid #eee; text-align: center;">
                    Thao Tác
                </div>
                """, unsafe_allow_html=True)
            
            for idx, item_id in enumerate(list(cart_items.keys())):
                col_item, col_qty, col_btn = st.columns([6, 2.5, 1.5])
                with col_item:
                    st.markdown(render_cart_item(item_id, products), unsafe_allow_html=True)
                with col_qty:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    new_qty = st.number_input("Số lượng", min_value=1, max_value=99, value=cart_items[item_id], key=f"qty_{item_id}_{idx}", label_visibility="collapsed")
                    if new_qty != cart_items[item_id]:
                        st.session_state['cart'][item_id] = new_qty
                        st.rerun()
                with col_btn:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("❌ Xóa", key=f"remove_cart_page_{item_id}_{idx}", use_container_width=True):
                        del st.session_state['cart'][item_id]
                        st.toast(f"Đã xóa khỏi giỏ hàng!")
                        time.sleep(0.5)
                        st.rerun()
            
            # Calculate total
            total_price = 0
            for item_id, qty in cart_items.items():
                product_row = products[products['item_id'] == item_id]
                if not product_row.empty:
                    total_price += product_row.iloc[0].get('price', 0) * qty
            
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; background: #fff; padding: 20px; border-radius: 8px; margin-top: 15px; border: 1px solid #fce4ec;">
                <div style="font-size: 16px; margin-right: 20px;">Tổng thanh toán ({total_items_count} Sản phẩm):</div>
                <div style="font-size: 24px; color: #ee4d2d; font-weight: 700;">₫{int(total_price):,}</div>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)
            
            st.markdown("<div class='section-header' style='margin-top: 40px;'>✨ GỢI Ý THÊM CHO GIỎ HÀNG</div>", unsafe_allow_html=True)
            with st.spinner("Đang tìm các sản phẩm phù hợp..."):
                cart_recs = model.cart_recommendations(list(cart_items.keys()), top_n=top_n)
            
            if cart_recs:
                rec_item_ids = [item_id for item_id, score in cart_recs]
                display_product_grid(rec_item_ids, prefix="cart_recs")
            else:
                st.info("Không có gợi ý nào cho các sản phẩm trong giỏ hàng hiện tại.")

    # -------------------------------------------------------------
    # PAGE: DANH SÁCH YÊU THÍCH
    # -------------------------------------------------------------
    elif active_menu == "💖 Yêu thích":
        st.markdown("<div class='section-header'>💖 SẢN PHẨM BẠN ĐÃ THÍCH</div>", unsafe_allow_html=True)
        if not selected_user:
            st.warning("Vui lòng **Đăng nhập** để xem danh sách sản phẩm yêu thích của bạn!")
        else:
            user_likes = st.session_state['interactions']
            # Find items with rating > 0 liked by user
            liked_items_df = user_likes[(user_likes['user_id'] == selected_user) & (user_likes['rating'] > 0)]
            if liked_items_df.empty:
                st.info("Bạn chưa có sản phẩm yêu thích nào. Hãy khám phá và thả tim cho các sản phẩm nhé!")
            else:
                liked_item_ids = liked_items_df['item_id'].tolist()[::-1]
                
                # Header row like Shopee/Cart
                col_h1, col_h2 = st.columns([7.5, 2.5])
                with col_h1:
                    st.markdown("""
                    <div style="display: flex; padding: 10px 15px; background: #fff; border-radius: 8px; margin-bottom: 10px; font-weight: 600; font-size: 13px; color: #888; border: 1px solid #eee;">
                        <div style="flex-grow: 1; padding-left: 85px;">Sản Phẩm</div>
                        <div style="width: 120px; text-align: right; margin-right: 10px;">Đơn Giá</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_h2:
                    st.markdown("""
                    <div style="padding: 10px 0; background: #fff; border-radius: 8px; margin-bottom: 10px; font-weight: 600; font-size: 13px; color: #888; border: 1px solid #eee; text-align: center;">
                        Thao Tác
                    </div>
                    """, unsafe_allow_html=True)
                
                for idx, item_id in enumerate(liked_item_ids):
                    col_item, col_btn = st.columns([7.5, 2.5])
                    with col_item:
                        st.markdown(render_cart_item(item_id, products), unsafe_allow_html=True)
                    with col_btn:
                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                        if st.button("💔 Bỏ thích", key=f"remove_fav_{item_id}_{idx}", use_container_width=True):
                            # Remove from session state
                            st.session_state['interactions'] = user_likes[
                                ~((user_likes['user_id'] == selected_user) & (user_likes['item_id'] == item_id))
                            ]
                            # Append a rating of 0.0 to custom_interactions.csv to override the 5.0 rating on next load
                            save_custom_interaction(selected_user, item_id, 0.0)
                            st.toast(f"Đã bỏ thích sản phẩm!")
                            time.sleep(0.5)
                            st.rerun()

                # Recommendation part
                st.markdown("<div class='section-header' style='margin-top: 40px;'>✨ GỢI Ý TƯƠNG TỰ VÀ LIÊN QUAN</div>", unsafe_allow_html=True)
                with st.spinner("Đang tìm các sản phẩm phù hợp..."):
                    fav_recs = model.cart_recommendations(liked_item_ids, top_n=top_n)
                
                if fav_recs:
                    rec_item_ids = [item_id for item_id, score in fav_recs]
                    display_product_grid(rec_item_ids, prefix="fav_recs")
                else:
                    st.info("Không có gợi ý nào cho các sản phẩm yêu thích hiện tại.")

    # -------------------------------------------------------------
    # PAGE: GỢI Ý CÁ NHÂN HÓA
    # -------------------------------------------------------------
    elif active_menu == "🎯 Gợi ý cho bạn":
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
            if not selected_user:
                st.warning("Vui lòng **Đăng nhập** và nhấn **Thích** một số sản phẩm để AI có thể đưa ra gợi ý cá nhân hóa cho riêng bạn!")
            else:
                st.info("Hãy đánh dấu 'Thích' thêm một số sản phẩm để AI có thể gợi ý cho bạn nhé!")


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
    # PAGE: CÀI ĐẶT AI & THÔNG SỐ
    # -------------------------------------------------------------
    elif active_menu == "⚙️ Cài đặt AI":
        st.markdown("<div class='section-header'>⚙️ CÀI ĐẶT THUẬT TOÁN AI</div>", unsafe_allow_html=True)
        st.markdown("Tinh chỉnh cách hệ thống AI gợi ý sản phẩm và nguồn dữ liệu đang sử dụng.")
        
        # Metrics Cards
        st.markdown("<br>", unsafe_allow_html=True)
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
                    <h3 style="color: #4c1d95;">{selected_user[:10] if selected_user else 'Khách'}</h3>
                    <span style="color: #8b5cf6;">Tài khoản</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<hr style='border: 0; height: 1px; background: rgba(233,30,99,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 👤 Tài khoản hiện tại")
            if selected_user:
                st.success(f"Bạn đang đăng nhập với tư cách: **{selected_user}**")
            else:
                st.warning("Bạn đang duyệt dưới tư cách khách (Chưa đăng nhập)")

        with c2:
            st.markdown("### 🎛️ Trọng số mô hình lai")
            st.session_state['top_n'] = st.slider("Số lượng gợi ý hiển thị", 8, 200, st.session_state['top_n'])
            
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
