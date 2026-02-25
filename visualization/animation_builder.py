"""
Trajectory Animation Builder
=============================
Stitches per-step PNG frames from the `frames/` directory into a GIF
animation saved to `animations/trajectory.gif`.

Uses PIL (Pillow) only — no ffmpeg or external binaries required.
"""

from __future__ import annotations

import os
import glob
from PIL import Image


class AnimationBuilder:
    """
    Builds GIF animations from sequentially-numbered PNG frames.
    """

    @staticmethod
    def build_gif(
        frames_dir: str,
        output_dir: str,
        filename: str = "trajectory.gif",
        fps: int = 10,
        max_frames: int = 0,
    ) -> str:
        """
        Stitch numbered PNGs into a GIF.

        Parameters
        ----------
        frames_dir : path containing XXXX.png files
        output_dir : destination for the GIF
        filename   : output filename (default trajectory.gif)
        fps        : playback speed in frames per second
        max_frames : if > 0, subsample to at most this many frames

        Returns
        -------
        str : absolute path to the generated GIF, or empty string on failure
        """
        os.makedirs(output_dir, exist_ok=True)

        # Collect and sort frame files
        pattern = os.path.join(frames_dir, "*.png")
        frame_paths = sorted(glob.glob(pattern))

        if not frame_paths:
            print("[AnimationBuilder] No frames found — skipping GIF generation.")
            return ""

        # Subsample if too many frames
        if max_frames > 0 and len(frame_paths) > max_frames:
            step = len(frame_paths) / max_frames
            frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

        # Load frames
        frames = []
        for fp in frame_paths:
            try:
                img = Image.open(fp)
                # Convert to RGB (remove alpha) for GIF compatibility
                if img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(img)
            except Exception:
                continue

        if len(frames) < 2:
            print("[AnimationBuilder] Insufficient frames for animation.")
            return ""

        output_path = os.path.join(output_dir, filename)
        duration_ms = max(20, int(1000 / fps))  # minimum 20ms per frame

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,         # infinite loop
            optimize=True,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[AnimationBuilder] Saved {filename} — {len(frames)} frames, "
              f"{fps} fps, {size_mb:.1f} MB")
        return output_path
