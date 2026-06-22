from sklearn.model_selection import train_test_split

from data_loader import get_products_and_interactions
from recommender import HybridRecommender


def split_interactions(interactions, test_size=0.2, random_state=42):
    if len(interactions) < 10:
        return interactions.copy(), interactions.copy()
    train, test = train_test_split(
        interactions,
        test_size=test_size,
        stratify=interactions['user_id'],
        random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main():
    products, interactions = get_products_and_interactions()
    print('Loaded products:', len(products))
    print('Loaded interactions:', len(interactions))

    train_interactions, test_interactions = split_interactions(interactions)
    print('Train interactions:', len(train_interactions))
    print('Test interactions:', len(test_interactions))

    model = HybridRecommender(products, train_interactions)
    print(model.train_collaborative())

    if len(products) == 0 or len(train_interactions) == 0:
        print('Không có đủ dữ liệu để tạo gợi ý.')
        return

    sample_user = train_interactions['user_id'].iloc[0]
    sample_item = products['item_id'].iloc[0]

    print('\n--- Demo Top-N Outputs ---')
    print('Similar Products for', sample_item)
    for item_id, score in model.similar_products(sample_item, top_n=5):
        print(' ', item_id, f'(score={score:.3f})')

    print('\nFrequently Bought Together for', sample_item)
    for item_id, count in model.frequently_bought_together(sample_item, top_n=5):
        print(' ', item_id, f'(count={count})')

    print(f'\nPersonalized Recommendations for user {sample_user}')
    for item_id, score, reasons in model.personalized_recommendations(sample_user, top_n=5):
        print(' ', item_id, f'(score={score:.3f}, reasons={reasons})')

    print('\nTrending Products')
    for item_id, count in model.trending_products(top_n=5):
        print(' ', item_id, f'(popularity={count})')

    sample_cart = [sample_item]
    print('\nCart Recommendations for cart', sample_cart)
    for item_id, score in model.cart_recommendations(sample_cart, top_n=5):
        print(' ', item_id, f'(score={score:.3f})')


if __name__ == '__main__':
    main()
