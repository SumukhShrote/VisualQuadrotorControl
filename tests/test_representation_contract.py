"""CPU tests for the maintained representation model and loss contract."""
import torch
from networks.temporal_transformer import TemporalEstimator


def test_visibility_feature_contract():
    x = torch.rand(4, 32, 8)
    x[0, -1, :2] = -1
    for flag in (False, True):
        model = TemporalEstimator(include_visibility=flag).eval()
        assert model.geometry.input_projection.in_features == (12 if flag else 8)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (4, 16)
        assert torch.isfinite(output).all()
        assert model.linear_velocity.window == 32
        assert model.geometry.window == model.angular_velocity.window == 8


if __name__ == "__main__":
    test_visibility_feature_contract()
    print("[PASS] Representation contracts")
