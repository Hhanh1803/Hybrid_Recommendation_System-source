import gzip
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RETAIL_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
RETAIL_FILE = DATA_DIR / "Online Retail.xlsx"

AMAZON_URLS = {
    "Digital_Music": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz",
    "Electronics": "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
}
AMAZON_DIR = DATA_DIR / "amazon"
AMAZON_DIR.mkdir(parents=True, exist_ok=True)


def download_retail_dataset(url: str = RETAIL_URL, target_path: Path | str = None) -> Path:
    target_path = Path(target_path) if target_path else RETAIL_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return target_path

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def load_retail_dataset(local_path: str | Path = None) -> pd.DataFrame:
    path = Path(local_path) if local_path else RETAIL_FILE
    if not path.exists():
        download_retail_dataset(target_path=path)
    return pd.read_excel(path, sheet_name=0, engine="openpyxl")


def download_amazon_dataset(category: str = "Digital_Music") -> Path:
    if category not in AMAZON_URLS:
        raise ValueError(f"Unknown Amazon category: {category}")
    target_path = AMAZON_DIR / f"{category}.json.gz"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return target_path

    response = requests.get(AMAZON_URLS[category], timeout=120)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def load_amazon_dataset(category: str = "Digital_Music", max_reviews: int = 50000) -> pd.DataFrame:
    path = download_amazon_dataset(category=category)
    rows = []
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_reviews and idx >= max_reviews:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def prepare_retail_data(df: pd.DataFrame, min_item_support: int = 20, min_user_support: int = 5):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    required = {'CustomerID', 'StockCode', 'Description', 'Quantity'}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset thiếu cột yêu cầu: {required - set(df.columns)}")

    df = df[df['Quantity'] > 0].dropna(subset=['CustomerID', 'Description'])
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['StockCode'] = df['StockCode'].astype(str)
    df['Description'] = df['Description'].astype(str)

    interactions = (
        df.groupby(['CustomerID', 'StockCode'], as_index=False)['Quantity']
        .sum()
        .rename(columns={'Quantity': 'quantity'})
    )
    interactions['rating'] = interactions['quantity'].apply(lambda q: min(5, max(1, int(np.log1p(q) + 1))))

    item_counts = interactions['StockCode'].value_counts()
    user_counts = interactions['CustomerID'].value_counts()
    active_items = item_counts[item_counts >= min_item_support].index
    active_users = user_counts[user_counts >= min_user_support].index

    interactions = interactions[
        interactions['StockCode'].isin(active_items) & interactions['CustomerID'].isin(active_users)
    ]

    products = (
        df[['StockCode', 'Description']]
        .drop_duplicates(subset=['StockCode'])
        .rename(columns={'StockCode': 'item_id', 'Description': 'description'})
    )
    products = products[products['item_id'].isin(interactions['StockCode'])].copy()
    products['name'] = products['description'].str[:80]

    # Deterministic price based on StockCode hash
    def get_retail_price(item_id):
        np.random.seed(hash(str(item_id)) % 2**32)
        return float(np.random.randint(20, 300) * 1000)

    # Deterministic rating based on StockCode hash
    def get_retail_rating(item_id):
        np.random.seed(hash(str(item_id)) % 2**32)
        return round(float(np.random.uniform(3.5, 5.0)), 1)

    # Infer category based on description keywords
    def get_retail_category(desc):
        desc = str(desc).upper()
        if 'BAG' in desc: return 'Bags'
        if 'LIGHT' in desc or 'LAMP' in desc: return 'Lighting'
        if 'BOTTLE' in desc or 'MUG' in desc or 'CUP' in desc or 'PLATE' in desc: return 'Kitchen'
        if 'CLOCK' in desc: return 'Home Decor'
        if 'CARD' in desc or 'WRAP' in desc: return 'Stationery'
        if 'TOY' in desc or 'DOLL' in desc: return 'Toys'
        return 'Home & Living'

    products['price'] = products['item_id'].apply(get_retail_price)
    products['rating'] = products['item_id'].apply(get_retail_rating)
    products['image_url'] = None
    products['category'] = products['description'].apply(get_retail_category)

    interactions = (
        interactions.rename(columns={'CustomerID': 'user_id', 'StockCode': 'item_id'})
        [['user_id', 'item_id', 'rating']]
    )
    return products.reset_index(drop=True), interactions.reset_index(drop=True)


def prepare_amazon_data(df: pd.DataFrame, min_item_support: int = 15, min_user_support: int = 3, max_items: int = 2000, category_name: str = "Digital_Music"):
    df = df.copy()
    required = {'reviewerID', 'asin', 'overall'}
    if not required.issubset(df.columns):
        raise ValueError(f"Amazon dataset thiếu cột yêu cầu: {required - set(df.columns)}")

    df = df.dropna(subset=['reviewerID', 'asin', 'overall'])
    df['reviewerID'] = df['reviewerID'].astype(str)
    df['asin'] = df['asin'].astype(str)
    df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
    df = df[df['overall'].notna()]

    for col in ['summary', 'reviewText']:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('').astype(str)
    df['description'] = (df['summary'].fillna('') + ' ' + df['reviewText'].fillna('')).str.strip()
    df.loc[df['description'] == '', 'description'] = 'No description available.'

    interactions = (
        df.groupby(['reviewerID', 'asin'], as_index=False)
        .agg(rating=('overall', 'mean'), description=('description', 'first'))
    )
    interactions['rating'] = interactions['rating'].clip(lower=1, upper=5)

    item_counts = interactions['asin'].value_counts()
    user_counts = interactions['reviewerID'].value_counts()
    active_items = item_counts[item_counts >= min_item_support].index
    active_users = user_counts[user_counts >= min_user_support].index

    interactions = interactions[
        interactions['asin'].isin(active_items) & interactions['reviewerID'].isin(active_users)
    ]

    top_items = item_counts[item_counts >= min_item_support].index
    if len(top_items) > max_items:
        selected_items = item_counts.loc[top_items].head(max_items).index
        interactions = interactions[interactions['asin'].isin(selected_items)]

    products = (
        interactions[['asin', 'description']]
        .drop_duplicates(subset=['asin'])
        .rename(columns={'asin': 'item_id', 'description': 'description'})
    )
    products['name'] = products['description'].str[:80]

    # Deterministic price based on asin hash
    def get_amazon_price(item_id):
        np.random.seed(hash(str(item_id)) % 2**32)
        return float(np.random.randint(10, 150) * 10000)

    # Real ratings from data mean
    avg_ratings = df.groupby('asin')['overall'].mean().to_dict()

    products['price'] = products['item_id'].apply(get_amazon_price)
    products['rating'] = products['item_id'].map(avg_ratings).fillna(4.0).round(1)
    products['image_url'] = None
    products['category'] = str(category_name).replace('_', ' ')

    interactions = interactions.rename(columns={'reviewerID': 'user_id', 'asin': 'item_id'})[['user_id', 'item_id', 'rating']]
    return products.reset_index(drop=True), interactions.reset_index(drop=True)


def load_kaggle_product_dataset(local_path: str | Path = None) -> pd.DataFrame:
    path = Path(local_path) if local_path else DATA_DIR / 'flipkart_com-ecommerce_sample.csv'
    if not path.exists():
        raise FileNotFoundError(f'Không tìm thấy Kaggle product dataset tại {path}')
    df = pd.read_csv(path)
    return df


def prepare_kaggle_products(df: pd.DataFrame, min_item_support: int = 10, num_synthetic_users: int = 2500):
    df = df.copy()
    if 'pid' not in df.columns or 'product_name' not in df.columns:
        raise ValueError('Kaggle product dataset thiếu cột pid hoặc product_name.')

    df['item_id'] = df['pid'].astype(str)
    df['name'] = df['product_name'].astype(str)
    if 'description' not in df.columns:
        df['description'] = df['product_category_tree'].astype(str)
    else:
        df['description'] = df['description'].fillna('').astype(str)

    # Extract first image URL
    def extract_first_image(val):
        if pd.isna(val):
            return None
        if isinstance(val, list):
            return val[0] if val else None
        text = str(val).strip()
        if text.startswith('[') and 'http' in text:
            urls = [part.strip(' "\'') for part in text[1:-1].split(',') if part.strip().strip(' "\'').startswith('http')]
            return urls[0] if urls else None
        if text.startswith('http'):
            return text
        return None

    # Clean price
    def clean_price(row):
        try:
            dp = float(row.get('discounted_price'))
            if dp > 0:
                return dp
        except Exception:
            pass
        try:
            rp = float(row.get('retail_price'))
            if rp > 0:
                return rp
        except Exception:
            pass
        np.random.seed(hash(row['item_id']) % 2**32)
        return float(np.random.randint(50, 500) * 1000)

    # Clean rating
    def clean_rating(row):
        try:
            rating_val = str(row.get('product_rating'))
            if rating_val and rating_val.strip().lower() != 'no rating available':
                r = float(rating_val)
                if 1.0 <= r <= 5.0:
                    return r
        except Exception:
            pass
        np.random.seed(hash(row['item_id']) % 2**32)
        return round(float(np.random.uniform(3.8, 4.9)), 1)

    # Clean category
    def clean_category(val):
        if pd.isna(val):
            return "General"
        text = str(val).strip()
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1].strip(' "\'')
        if '>>' in text:
            return text.split('>>')[0].strip(' "\'')
        return text if text else "General"

    df['image_url'] = df['image'].apply(extract_first_image)
    df['price'] = df.apply(clean_price, axis=1)
    df['rating'] = df.apply(clean_rating, axis=1)
    df['category'] = df['product_category_tree'].apply(clean_category)

    products = df[['item_id', 'name', 'description', 'image_url', 'price', 'rating', 'category']].drop_duplicates(subset=['item_id']).copy()
    products['name'] = products['name'].astype(str)
    products['description'] = products['description'].astype(str)

    # Build synthetic users by sampling products within the same category tree
    category_groups = df.groupby('category')['item_id'].apply(list).to_dict()

    synthetic_rows = []
    user_id = 1
    for category, items in category_groups.items():
        for i in range(3):
            chosen = list(set(np.random.choice(items, min(len(items), 15), replace=False)))
            for item in chosen:
                synthetic_rows.append({'user_id': f'KUSER_{user_id}', 'item_id': item, 'rating': float(np.random.choice([4, 5], p=[0.4, 0.6]))})
            user_id += 1

    # Add further random interactions for diversity
    all_items = products['item_id'].tolist()
    for i in range(num_synthetic_users):
        chosen_items = list(set(np.random.choice(all_items, min(8, len(all_items)), replace=False)))
        for item in chosen_items:
            synthetic_rows.append({'user_id': f'SYN_{i}', 'item_id': item, 'rating': float(np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))})

    interactions = pd.DataFrame(synthetic_rows)
    interactions = interactions.groupby(['user_id', 'item_id'], as_index=False)['rating'].mean()
    return products, interactions


def get_dummy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = pd.DataFrame({
        'item_id': ['P1', 'P2', 'P3', 'P4'],
        'name': ['iPhone 15', 'Samsung S24', 'MacBook Air', 'iPad Pro'],
        'category': ['Phone', 'Phone', 'Laptop', 'Tablet'],
        'description': ['Apple smartphone', 'Android flagship', 'Apple laptop M3', 'Apple tablet'],
        'price': [25000000.0, 20000000.0, 30000000.0, 18000000.0],
        'rating': [4.8, 4.7, 4.9, 4.6],
        'image_url': [None, None, None, None]
    })

    interactions = pd.DataFrame({
        'user_id': ['U1', 'U1', 'U2', 'U2', 'U3'],
        'item_id': ['P1', 'P3', 'P1', 'P2', 'P4'],
        'rating': [5, 4, 5, 4, 5]
    })
    return products, interactions


def load_electronics_dataset():
    path = DATA_DIR / 'electronics_product.csv'
    if not path.exists():
        raise FileNotFoundError(f'Không tìm thấy Electronics dataset tại {path}')
    df = pd.read_csv(path)
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
    return df


def prepare_electronics_data(df: pd.DataFrame, num_synthetic_users: int = 2000):
    df = df.copy()
    
    # Generate item_id
    df['item_id'] = ['ELEC_' + str(i).zfill(5) for i in range(len(df))]
    df['name'] = df['name'].astype(str)
    df['description'] = df['name'] + ' (' + df['sub_category'].astype(str) + ')'
    df['category'] = df['main_category'].astype(str)
    df['image_url'] = df['image'].astype(str)

    def clean_rating(val):
        try:
            r = float(str(val).replace(',', '.'))
            if 1.0 <= r <= 5.0:
                return r
        except Exception:
            pass
        return 4.5

    df['rating'] = df['ratings'].apply(clean_rating)
    
    def get_price(val):
        try:
            val = str(val).replace('₹', '').replace(',', '').strip()
            return float(val) * 300
        except Exception:
            return float(np.random.randint(1000, 20000) * 1000)
            
    df['price'] = df['actual_price'].apply(get_price)

    products = df[['item_id', 'name', 'description', 'image_url', 'price', 'rating', 'category']].copy()

    # Create synthetic users
    synthetic_rows = []
    all_items = products['item_id'].tolist()
    for i in range(num_synthetic_users):
        num_items = np.random.randint(3, 10)
        chosen_items = list(set(np.random.choice(all_items, min(num_items, len(all_items)), replace=False)))
        for item in chosen_items:
            rating_val = float(np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
            synthetic_rows.append({'user_id': f'ELEC_USER_{i}', 'item_id': item, 'rating': rating_val})

    interactions = pd.DataFrame(synthetic_rows)
    interactions = interactions.groupby(['user_id', 'item_id'], as_index=False)['rating'].mean()
    
    return products, interactions


def get_products_and_interactions(source: str = 'electronics', amazon_category: str = 'Digital_Music') -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = load_electronics_dataset()
        products, interactions = prepare_electronics_data(df)
        
        if products.empty or interactions.empty:
            raise RuntimeError('Dữ liệu không đủ sau bước tiền xử lý.')
        return products, interactions
    except Exception as exc:
        warnings.warn(f"Không thể tải dữ liệu online, chuyển sang tập dữ liệu mẫu. Chi tiết: {exc}")
        return get_dummy_data()
