from functools import lru_cache


@lru_cache(maxsize=1)
def load_xt_model():
    # Stub: Replace with actual model logic
    class DummyXTModel:
        def get_value(self, x, y):
            # Example: scale x from 0 to 1, y from 0 to 1 and multiply
            return round((x / 120) * (y / 80), 4)

    return DummyXTModel()