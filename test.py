from kegra.utils import load_karate_data
from kegra.utils import get_karate_splits
X, A, y = load_karate_data()
y, idx, train_mask = get_karate_splits(y)

print(y.shape[0])