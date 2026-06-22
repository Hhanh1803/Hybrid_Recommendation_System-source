import os
import json
import pandas as pd
import hashlib
import requests
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess


def get_product_image(product_name):
    seed = hashlib.md5(product_name.encode()).hexdigest()[:10]
    return f"https://picsum.photos/500/500?random={seed}"


def normalize_candidate_df(df: pd.DataFrame):
    if 'image_url' in df.columns:
        pass
    elif 'image' in df.columns:
        def extract_first_image(val):
            if pd.isna(val):
                return None
            if isinstance(val, list):
                return val[0] if val else None
            text = str(val)
            if text.startswith('[') and 'http' in text:
                urls = [part for part in text.replace('"', '').replace("'", '').split(',') if part.strip().startswith('http')]
                return urls[0].strip() if urls else None
            if text.startswith('http'):
                return text.strip()
            return None
        df['image_url'] = df['image'].apply(extract_first_image)
    elif 'images' in df.columns:
        df['image_url'] = df['images'].apply(lambda v: str(v).split(',')[0].strip() if pd.notna(v) else None)
    else:
        return None

    if 'product_name' in df.columns and 'name' not in df.columns:
        df = df.rename(columns={'product_name': 'name'})
    if 'product_title' in df.columns and 'name' not in df.columns:
        df = df.rename(columns={'product_title': 'name'})

    if 'name' not in df.columns or 'image_url' not in df.columns:
        return None

    df = df[['name', 'image_url']].copy()
    df = df[df['image_url'].notna() & df['name'].notna()]
    df['name'] = df['name'].astype(str)
    df['image_url'] = df['image_url'].astype(str)
    return df


def load_image_dataset():
    candidates = glob('data/*.csv')
    data_frames = []
    for p in candidates:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        normalized = normalize_candidate_df(df)
        if normalized is not None and not normalized.empty:
            normalized['source_file'] = os.path.basename(p)
            data_frames.append(normalized)
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    return None


def build_image_search_index(image_df):
    if image_df is None or image_df.empty:
        return None
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000)
    matrix = tfidf.fit_transform(image_df['name'].astype(str))
    return {'vectorizer': tfidf, 'matrix': matrix}


def find_best_image_by_name(product_name, image_df, index):
    if image_df is None or image_df.empty or index is None:
        return None
    query = index['vectorizer'].transform([product_name])
    sim = cosine_similarity(query, index['matrix']).flatten()
    best = sim.argmax()
    if sim[best] >= 0.20:
        return image_df.iloc[best]['image_url']
    return None


def load_image_cache(path="data/image_cache.json"):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_image_cache(cache, path="data/image_cache.json"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def get_unsplash_image(query, access_key, cache=None):
    if not query:
        return None
    key = query.lower()
    if cache is None:
        cache = load_image_cache()
    if key in cache:
        return cache[key]
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query.split()[0], "per_page": 1}
    headers = {"Accept-Version": "v1", "Authorization": f"Client-ID {access_key}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            results = data.get('results') or []
            if results:
                img = results[0].get('urls', {}).get('regular')
                if img:
                    cache[key] = img
                    save_image_cache(cache)
                    return img
    except Exception:
        pass
    return None


def validate_image_url(url, cache_path="data/image_url_cache.json"):
    """Check URL with HEAD and cache results to avoid repeated network calls."""
    try:
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        if url in cache:
            return cache[url]
        # try HEAD then GET if HEAD not allowed
        try:
            r = requests.head(url, timeout=5, allow_redirects=True)
            ok = (r.status_code == 200 and 'image' in (r.headers.get('content-type') or ''))
        except Exception:
            try:
                r = requests.get(url, timeout=5, stream=True)
                ok = (r.status_code == 200 and 'image' in (r.headers.get('content-type') or ''))
            except Exception:
                ok = False
        cache[url] = ok
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        except Exception:
            pass
        return ok
    except Exception:
        return False


def get_valid_image_for_name(name, image_df, index, unsplash_key=None):
    """Return a valid image URL: try dataset match, validate, then Unsplash, then Picsum."""
    if image_df is not None and index is not None:
        url = find_best_image_by_name(name, image_df, index)
        if url:
            if validate_image_url(url):
                return url
    # try exact dataset match
    if image_df is not None and 'name' in image_df.columns:
        exact = image_df[image_df['name'].str.lower() == str(name).lower()]
        if not exact.empty:
            url = exact.iloc[0]['image_url']
            if validate_image_url(url):
                return url

    # try Unsplash
    if unsplash_key:
        url = get_unsplash_image(name, unsplash_key)
        if url:
            return url

    # fallback picsum
    return get_product_image(name)


def check_kaggle_cli():
    try:
        res = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


def download_kaggle_dataset(dataset_slug, dest_dir="data"):
    if not check_kaggle_cli():
        return False, "kaggle CLI không được tìm thấy. Cài đặt kaggle CLI hoặc upload file dataset thủ công."
    os.makedirs(dest_dir, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", dest_dir, "--unzip"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            return True, f"Đã tải và giải nén vào {dest_dir}"
        else:
            return False, proc.stderr or proc.stdout
    except subprocess.TimeoutExpired:
        return False, "Download timed out"
    except Exception as e:
        return False, str(e)
