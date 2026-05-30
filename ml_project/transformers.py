from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder


class LocationTargetEncoder(BaseEstimator, TransformerMixin):
    """Target-encode the 'location' column using category_encoders."""

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.encoder = TargetEncoder(cols=["location"], smoothing=smoothing)

    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.encoder.get_feature_names_out(input_features)


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Calculates interaction features like sqft_per_room, room_density, etc."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["sqft_per_room"] = X["total_sqft"] / (X["bhk"] + X["bath"])
        X["room_density"] = X["bhk"] / (X["total_sqft"] / 1000)
        X["bath_to_bhk"] = X["bath"] / X["bhk"].apply(lambda x: max(x, 1))
        X["total_rooms"] = X["bhk"] + X["bath"] + X["balcony"]
        return X

    def get_feature_names_out(self, input_features=None):
        return list(input_features) + ["sqft_per_room", "room_density", "bath_to_bhk", "total_rooms"]
