
from pathlib import Path

import cv2
import numpy as np
import torch


class SideBySideVideoRecorder:
    def __init__(self, path, fps=50):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.width = 1440
        self.height = 500
        self.count = 0

        self.writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (self.width, self.height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.path}")

        print("[VIDEO] Recording:", self.path, flush=True)

    @staticmethod
    def to_bgr(image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)

        if image.ndim == 4:
            image = image[0]

        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise RuntimeError(
                f"Unexpected camera image shape: {image.shape}"
            )

        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size and image.max() <= 1.0:
                image *= 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def capture(self, env, state, step, action):
        # Render the same simulated state from both viewpoints.
        third_rgb = env.render()

        if third_rgb is None:
            raise RuntimeError(
                "Third-person viewport returned None. "
                "Check rgb_array rendering and camera launch flags."
            )

        base = env.unwrapped
        camera = getattr(base, "_camera", None)

        if camera is None:
            raise RuntimeError(
                "Onboard camera was not created."
            )

        onboard_rgb = camera.data.output.get("rgb")

        if onboard_rgb is None:
            if step == 0:
                print("[VIDEO] Waiting for first onboard frame")
                return
            raise RuntimeError("Onboard RGB output unavailable")

        third = self.to_bgr(third_rgb)
        onboard = self.to_bgr(onboard_rgb)

        if self.count == 0:
            print(
                "[VIDEO] Third-person source:",
                third.shape,
                "Onboard source:",
                onboard.shape,
                flush=True,
            )

        third = cv2.resize(
            third, (720, 405), interpolation=cv2.INTER_AREA
        )
        onboard = cv2.resize(
            onboard, (720, 405),
            interpolation=cv2.INTER_NEAREST,
        )

        canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )

        canvas[48:453, :720] = third
        canvas[48:453, 720:] = onboard

        cv2.putText(
            canvas, "THIRD-PERSON VIEW",
            (20, 33), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2,
        )

        cv2.putText(
            canvas, "ONBOARD CAMERA",
            (740, 33), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2,
        )

        s = state.detach().cpu().numpy()
        a = action.detach().cpu().numpy()

        pos = s[:3]
        vel = s[3:6]

        distance = float(np.linalg.norm(pos))
        speed = float(np.linalg.norm(vel))

        precise = distance < 0.10 and speed < 0.20

        status = "PRECISE" if precise else "OUTSIDE PRECISION"

        hud = (
            f"t={step/50:.2f}s   "
            f"position_error={distance:.3f}m   "
            f"velocity={speed:.3f}m/s   "
            f"{status}"
        )

        cv2.putText(
            canvas, hud, (16, 478),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1,
        )

        # Signed body-frame position errors.
        cv2.putText(
            canvas,
            f"Body error X={pos[0]:+.3f} "
            f"Y={pos[1]:+.3f} Z={pos[2]:+.3f} m",
            (755, 478),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47, (0, 220, 255), 1,
        )

        self.writer.write(canvas)

        if self.count == 0:
            snapshot = self.path.with_suffix(".png")
            cv2.imwrite(str(snapshot), canvas)
            print("[VIDEO] Preview:", snapshot, flush=True)

        self.count += 1

    def close(self):
        self.writer.release()
        print(
            f"[VIDEO SAVED] {self.path} frames={self.count}",
            flush=True,
        )
