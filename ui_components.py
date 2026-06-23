import streamlit as st
import pandas as pd
import numpy as np
import hashlib

def render_header():
    st.markdown("""
    <div class='header-banner'>
        <div class='header-title'>🛍️ ShopSmart</div>
        <div class='header-subtitle'>Khám phá sản phẩm được AI gợi ý dành cho bạn</div>
    </div>
    """, unsafe_allow_html=True)


def render_cart_item(item_id, products):
    try:
        matching_products = products[products['item_id'] == item_id]
        if matching_products.empty:
            return f"<div class='cart-item'>Sản phẩm {item_id} không tồn tại</div>"
            
        product = matching_products.iloc[0]
        name = product.get('name', str(item_id))
        
        image_url = product.get('image_url')
        if not image_url or pd.isna(image_url):
            seed = hashlib.md5(name.encode()).hexdigest()[:10]
            image_url = f"https://picsum.photos/100/100?random={seed}"

        price = product.get('price', 0)
        category = product.get('category', 'Chung')
        price_str = f"{int(price):,}".replace(",", ".")

        html = f"""
        <div class='cart-item'>
            <img class='cart-item-image' src="{image_url}" alt='{name}' />
            <div class='cart-item-info'>
                <div class='cart-item-name' title='{name}'>{name}</div>
                <div class='cart-item-category'>Phân loại: {category}</div>
            </div>
            <div class='cart-item-price'>₫{price_str}</div>
        </div>
        """
        return html
    except Exception as e:
        return f"<div class='cart-item'>Lỗi tải sản phẩm: {str(e)}</div>"


def render_product_card(item_id, products):
    try:
        # Check if item exists in products dataframe
        matching_products = products[products['item_id'] == item_id]
        if matching_products.empty:
            return f"<div class='product-card'><div class='product-info'>Sản phẩm {item_id} không tồn tại</div></div>"
            
        product = matching_products.iloc[0]
        name = product.get('name', str(item_id))
        
        # Retrieve image URL from products DataFrame
        image_url = product.get('image_url')
        if not image_url or pd.isna(image_url):
            # Fallback to Picsum random image based on name seed
            seed = hashlib.md5(name.encode()).hexdigest()[:10]
            image_url = f"https://picsum.photos/500/500?random={seed}"

        # Retrieve price, rating, category
        price = product.get('price', 0)
        rating = product.get('rating', 4.2)
        category = product.get('category', 'Chung')

        # Deterministic random factors based on item_id
        seed_val = int(hashlib.md5(str(item_id).encode()).hexdigest()[:8], 16)
        np.random.seed(seed_val)
        sold = np.random.randint(15, 1200)
        discount = np.random.randint(5, 50)
        locations = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Nước ngoài", "Hải Phòng"]
        location = locations[np.random.randint(0, len(locations))]

        # Star rating representation
        stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
        price_str = f"{int(price):,}".replace(",", ".")
        
        # old price is price + discount%
        old_price = int(price / (1 - discount/100))
        old_price_str = f"{old_price:,}".replace(",", ".")

        html = f"""
        <div class='product-card'>
            <div class='product-image'>
                <img src="{image_url}" alt='{name}' />
                <div class='discount-badge'>
                    <div class='discount-percent'>{discount}%</div>
                    <div class='discount-label'>GIẢM</div>
                </div>
                <div class='mall-badge'>Mall</div>
            </div>
            <div class='product-info'>
                <div class='product-name' title='{name}'>{name}</div>
                <div class='product-meta-row' style='margin-bottom: 5px; display: flex; justify-content: space-between;'>
                    <div class='product-rating' style='font-size: 0.75em;'>{stars} <span style='color: #888; font-size: 0.9em;'>({sold})</span></div>
                </div>
                <div class='product-price-row' style='display: flex; align-items: baseline; gap: 5px; margin-top: auto;'>
                    <div class='product-price' style='color: #e91e63; font-weight: 700; font-size: 1.1em;'><span class='currency' style='font-size: 0.8em; margin-right: 2px;'>₫</span>{price_str}</div>
                    <div style='color: #999; text-decoration: line-through; font-size: 0.8em;'>₫{old_price_str}</div>
                </div>
            </div>
        </div>
        """
        return html
    except Exception as e:
        return f"<div class='product-card'><div class='product-info'>Lỗi tải sản phẩm: {str(e)}</div></div>"

