from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch


class DroneFollowCornersVideoRecorder:
    def __init__(self, path, fps=50):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.width = 1440
        self.height = 500
        self.count = 0
        self.max_scene_std = 0.0
        self.closed = False

        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is unavailable")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "1440x500",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.path),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print("[VIDEO] H.264 recording:", self.path, flush=True)

    @staticmethod
    def to_bgr(image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)

        if image.ndim == 4:
            image = image[0]

        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise RuntimeError(
                f"Unexpected external-camera shape: {image.shape}"
            )

        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.max() <= 1.0:
                image *= 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        return cv2.cvtColor(
            image[:, :, :3],
            cv2.COLOR_RGB2BGR,
        )

    @staticmethod
    def corner_panel(corner_history):
        panel = np.full(
            (405, 720, 3),
            22,
            dtype=np.uint8,
        )

        # The controller receives 32 frames x 8 corner values.
        # Display the most recent four (x, y) corner pairs.
        values = (
            corner_history.detach()
            .float()
            .cpu()
            .reshape(32, 8)[-1]
            .numpy()
        )

        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite marker corners")

        corners = values.reshape(4, 2)

        valid = bool(
            np.all((corners >= 0) & (corners <= 1))
            and cv2.contourArea(
                corners.astype(np.float32)
            ) > 1e-7
        )

        # Keep the image square to preserve marker geometry.
        size = 405
        x_offset = (720 - size) // 2

        cv2.putText(
            panel,
            "Latest ArUco corner observation",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
        )

        cv2.rectangle(
            panel,
            (x_offset, 0),
            (x_offset + size - 1, size - 1),
            (80, 80, 80),
            1,
        )

        if valid:
            pixels = np.column_stack([
                x_offset + corners[:, 0] * (size - 1),
                corners[:, 1] * (size - 1),
            ]).round().astype(np.int32)

            cv2.polylines(
                panel,
                [pixels],
                True,
                (70, 170, 70),
                1,
                cv2.LINE_AA,
            )

            for index, (x, y) in enumerate(pixels):
                x, y = int(x), int(y)

                cv2.circle(
                    panel,
                    (x, y),
                    6,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    panel,
                    str(index),
                    (x + 9, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
        else:
            cv2.putText(
                panel,
                "No complete valid marker observation",
                (140, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                1,
            )

        return panel

    def capture(self, env, state, step, action, corners):
        if self.closed:
            raise RuntimeError("Recorder already closed")

        # Restore the original simulator viewport.
        # The camera is NOT moved to a new prim.

        # Follow the physical drone in the simulation.
        # The camera is behind, above, and beside it.
        base = env.unwrapped

        drone = (
            base._robot.data.root_pos_w[0]
            .detach().cpu().tolist()
        )

        hover = (
            base._hover_point_w[0]
            .detach().cpu().tolist()
        )

        px, py, pz = [float(x) for x in drone]
        hx, hy, hz = [float(x) for x in hover]

        eye = (
            px - 0.55,
            py - 0.40,
            pz + 0.30,
        )

        lookat = (
            px,
            py,
            pz,
        )

        base.sim.set_camera_view(
            eye=eye,
            target=lookat,
        )

        if self.count in (0, 100, 250):
            print(
                "[FOLLOW CAMERA]",
                f"frame={self.count}",
                f"drone={drone}",
                f"target={hover}",
                f"eye={eye}",
                flush=True,
            )

        rgb = env.render()

        if rgb is None:
            raise RuntimeError(
                "Simulator viewport returned no RGB image"
            )

        third = self.to_bgr(rgb)

        third = cv2.resize(
            third,
            (720, 405),
            interpolation=cv2.INTER_AREA,
        )

        scene_std = float(
            third.astype(np.float32)
            .std(axis=(0, 1))
            .mean()
        )

        self.max_scene_std = max(
            self.max_scene_std,
            scene_std,
        )

        right = self.corner_panel(corners)

        canvas = np.zeros(
            (500, 1440, 3),
            dtype=np.uint8,
        )

        canvas[48:453, :720] = third
        canvas[48:453, 720:] = right

        # Top-down position indicator.
        # Uses ground truth only for visualization.
        # Never passed to the deployed policy.

        map_left = 15
        map_top = 260
        map_right = 215
        map_bottom = 444

        cv2.rectangle(
            canvas,
            (map_left, map_top),
            (map_right, map_bottom),
            (18, 18, 18),
            -1,
        )

        cv2.rectangle(
            canvas,
            (map_left, map_top),
            (map_right, map_bottom),
            (180, 180, 180),
            1,
        )

        cv2.putText(
            canvas,
            "TOP VIEW (WORLD XY)",
            (24, 279),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
        )

        cx, cy = 115, 358
        scale = 240.0  # pixels per meter

        # Hover target is always at the center.
        cv2.drawMarker(
            canvas,
            (cx, cy),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            16,
            2,
        )

        dx = px - hx
        dy = py - hy

        drone_x = int(cx + scale * dx)
        drone_y = int(cy - scale * dy)

        drone_x = max(23, min(207, drone_x))
        drone_y = max(290, min(435, drone_y))

        cv2.line(
            canvas,
            (cx, cy),
            (drone_x, drone_y),
            (0, 180, 255),
            1,
        )

        cv2.circle(
            canvas,
            (drone_x, drone_y),
            7,
            (0, 100, 255),
            -1,
        )

        cv2.putText(
            canvas,
            f"XY error: {(dx*dx+dy*dy)**0.5:.3f} m",
            (25, 428),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )


        cv2.putText(
            canvas,
            "THIRD-PERSON CAMERA",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            canvas,
            "POLICY CORNER OBSERVATIONS",
            (740, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        state = state.detach().cpu().numpy()

        position_error = float(
            np.linalg.norm(state[:3])
        )

        speed = float(
            np.linalg.norm(state[3:6])
        )

        precise = (
            position_error < 0.10
            and speed < 0.20
        )

        text = (
            f"t={step / 50:.2f}s  "
            f"position={position_error:.3f}m  "
            f"speed={speed:.3f}m/s  "
            f"{'PRECISE' if precise else 'OUTSIDE PRECISION'}"
        )

        cv2.putText(
            canvas,
            text,
            (16, 478),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            1,
        )

        # Save separate previews rather than overwriting t=0.
        if self.count in (0, 25, 100, 250, 400):
            snapshot = self.path.with_name(
                f"{self.path.stem}_frame_{self.count:03d}.png"
            )

            cv2.imwrite(str(snapshot), canvas)

            if self.count >= 100:
                cv2.imwrite(
                    str(self.path.with_suffix(".png")),
                    canvas,
                )

        try:
            self.process.stdin.write(canvas.tobytes())
        except BrokenPipeError as exc:
            error = self.process.stderr.read().decode(
                errors="replace"
            )
            raise RuntimeError(
                f"FFmpeg stopped unexpectedly: {error}"
            ) from exc

        self.count += 1

        if self.count == 50:
            print(
                "[THIRD-PERSON] max_pixel_std="
                f"{self.max_scene_std:.2f}",
                flush=True,
            )

            if self.max_scene_std < 3.0:
                raise RuntimeError(
                    "Third-person sensor is still blank. "
                    "Do not batch-record."
                )

    def close(self):
        if self.closed:
            return

        self.closed = True
        self.process.stdin.close()

        code = self.process.wait()

        error = self.process.stderr.read().decode(
            errors="replace"
        )
        self.process.stderr.close()

        print(
            f"[VIDEO SAVED] {self.path} "
            f"frames={self.count} encoder_exit={code}",
            flush=True,
        )

        if code != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed: {error}"
            )
