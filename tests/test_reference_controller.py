"""Validate the retained reference controller without Isaac Sim."""

import torch

from utils.checkpoint_utils import (
    verify_reference_artifacts,
    load_reference_controller,
    load_deployment_policy,
)


def test_reference_artifacts():
    assert verify_reference_artifacts() == 12


def test_reference_policy_matches_export():
    model = load_reference_controller(device="cpu").eval()
    scripted = load_deployment_policy(device="cpu").eval()

    generator = torch.Generator().manual_seed(4300)

    corners = torch.rand(32, 256, generator=generator)
    actions = torch.rand(32, 8, 4, generator=generator) * 2 - 1

    with torch.no_grad():
        output = model(corners, actions)
        expected = scripted(corners, actions)

    assert output.shape == (32, 4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, expected, atol=1e-6, rtol=1e-6
    )


if __name__ == "__main__":
    test_reference_artifacts()
    test_reference_policy_matches_export()
    print("[PASS] Reference controller integrity and equivalence")
