from common.data import load_splits


def test_splits_shapes_and_no_leakage():
    splits = load_splits(cache=False)

    assert splits.n_features == 13
    assert splits.n_classes == 114

    total = splits.X_train.shape[0] + splits.X_val.shape[0] + splits.X_test.shape[0]
    assert total > 100_000

    # Roughly 80/10/10 as configured in load_splits().
    assert 0.75 < splits.X_train.shape[0] / total < 0.85
    assert 0.05 < splits.X_val.shape[0] / total < 0.15
    assert 0.05 < splits.X_test.shape[0] / total < 0.15


def test_scaling_fit_on_train_only():
    splits = load_splits(cache=False)
    # Train features should be ~standardized (mean 0, std 1) since the
    # scaler was fit on them directly.
    assert abs(splits.X_train.mean()) < 0.1
    assert abs(splits.X_train.std() - 1.0) < 0.1


def test_class_names_cover_all_labels():
    splits = load_splits(cache=False)
    assert splits.y_train.max() < splits.n_classes
    assert splits.y_train.min() >= 0
