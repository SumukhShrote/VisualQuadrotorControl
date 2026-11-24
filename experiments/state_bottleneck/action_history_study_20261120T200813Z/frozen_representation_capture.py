
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path("/home/sumukh/AerialCatching")
BASE = ROOT / "experiments/state_bottleneck"

CKPT = BASE / "long_horizon_gae_20261123T032419Z/model_9.pt"
SOURCE = BASE / "launch_action_history_study.py"


class FrozenRepresentationCapture(nn.Module):
    def __init__(self, scripted_policy_path, device):
        super().__init__()

        spec = importlib.util.spec_from_file_location(
            "frozen_probe_policy_source",
            SOURCE,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load policy source")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.actor = module.Policy(8)

        ckpt = torch.load(
            CKPT,
            map_location="cpu",
            weights_only=False,
        )

        weights = {
            name.removeprefix("actor."): value
            for name, value in ckpt["model_state_dict"].items()
            if name.startswith("actor.")
        }

        self.actor.load_state_dict(weights, strict=True)
        self.actor = self.actor.to(device).eval()

        for p in self.actor.parameters():
            p.requires_grad_(False)

        self.reference = torch.jit.load(
            scripted_policy_path,
            map_location=device,
        ).eval()

        self.checked = False
        self.last_features = None

        # Capture the ACTUAL input to the policy head.
        # First 192 dimensions are visual features.
        # Last 32 dimensions are previous actions.
        self.actor.head[0].register_forward_pre_hook(
            self._capture_head_input
        )

        self.storage = {
            "features": [],
            "corners": [],
            "actions8": [],
            "teacher_state": [],
            "commands": [],
            "steps": [],
            "env_ids": [],
        }

        print("[PROBE] Frozen checkpoint loaded", flush=True)

    def _capture_head_input(self, module, inputs):
        x = inputs[0].detach()

        if x.ndim != 2 or x.shape[1] != 224:
            raise RuntimeError(
                f"Expected 224-D policy-head input, got {x.shape}"
            )

        self.last_features = x

    @torch.no_grad()
    def forward(self, corners, actions8):
        result = self.actor(corners, actions8)

        if not self.checked:
            reference_result = self.reference(
                corners,
                actions8,
            )

            difference = (
                result - reference_result
            ).abs().max().item()

            print(
                f"[POLICY MATCH] max_action_error={difference:.9g}",
                flush=True,
            )

            if difference > 2e-5:
                raise RuntimeError(
                    "Python policy does not match TorchScript. "
                    "Aborting diagnostic."
                )

            self.checked = True
            self.reference = None

        return result

    @torch.no_grad()
    def capture(
        self,
        corners,
        actions8,
        teacher_state,
        live,
        step,
        commands,
    ):
        # Collect every second control step: 25 Hz.
        # We retain the full 32-step visual history.
        if step % 2:
            return

        indices = np.flatnonzero(live)

        if len(indices) == 0:
            return

        ids = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=corners.device,
        )

        values = {
            "features": self.last_features[ids, :192],
            "corners": corners[ids],
            "actions8": actions8[ids].reshape(-1, 32),
            "teacher_state": teacher_state[ids, :9],
            "commands": commands[ids],
        }

        for name, tensor in values.items():
            self.storage[name].append(
                tensor.detach().float().cpu().numpy()
            )

        self.storage["steps"].append(
            np.full(len(indices), step, dtype=np.int32)
        )

        self.storage["env_ids"].append(
            indices.astype(np.int32)
        )

    def save(self, path):
        data = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in self.storage.items()
            if chunks
        }

        if not data:
            raise RuntimeError("No diagnostic data collected")

        n = len(data["features"])

        assert data["features"].shape == (n, 192)
        assert data["corners"].shape == (n, 256)
        assert data["actions8"].shape == (n, 32)
        assert data["teacher_state"].shape == (n, 9)

        np.savez_compressed(path, **data)

        print(
            f"[PROBE SAVED] {path} samples={n}",
            flush=True,
        )
