"""
Trajectory Animation Builder
=============================
Stitches per-step PNG frames from the `frames/` directory into a smooth
MP4 video saved to `animations/trajectory.mp4`.

Uses OpenCV (cv2) for portable MP4 encoding — no external ffmpeg binary
required. Falls back to Pillow GIF if cv2 is unavailable.
"""

from __future__ import annotations

import os
import glob


class AnimationBuilder:
    """
    Builds trajectory animations from sequentially-numbered PNG frames.
    """

    @staticmethod
    def build_mp4(
        frames_dir: str,
        output_dir: str,
        filename: str = "trajectory.mp4",
        fps: int = 4,
        max_frames: int = 0,
    ) -> str:
        """
        Stitch numbered PNGs into an MP4 video.

        Parameters
        ----------
        frames_dir : path containing XXXX.png files
        output_dir : destination for the MP4
        filename   : output filename (default trajectory.mp4)
        fps        : playback speed in frames per second (lower = slower)
        max_frames : if > 0, subsample to at most this many frames

        Returns
        -------
        str : absolute path to the generated MP4, or empty string on failure
        """
        os.makedirs(output_dir, exist_ok=True)

        # Collect and sort frame files
        pattern = os.path.join(frames_dir, "*.png")
        frame_paths = sorted(glob.glob(pattern))

        if not frame_paths:
            print("[AnimationBuilder] No frames found — skipping animation.")
            return ""

        # Subsample if too many frames
        if max_frames > 0 and len(frame_paths) > max_frames:
            step = len(frame_paths) / max_frames
            frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

        output_path = os.path.join(output_dir, filename)

        try:
            import cv2
            return AnimationBuilder._write_mp4_cv2(
                frame_paths, output_path, fps
            )
        except ImportError:
            print("[AnimationBuilder] cv2 unavailable — falling back to GIF.")
            return AnimationBuilder._fallback_gif(
                frame_paths, output_dir, fps
            )

    @staticmethod
    def _write_mp4_cv2(frame_paths: list, output_path: str, fps: int) -> str:
        """Write MP4 using OpenCV VideoWriter."""
        import cv2

        # Read first frame to get dimensions
        sample = cv2.imread(frame_paths[0])
        if sample is None:
            print("[AnimationBuilder] Cannot read first frame.")
            return ""

        h, w = sample.shape[:2]
        # Ensure dimensions are even (required by H.264)
        w = w if w % 2 == 0 else w - 1
        h = h if h % 2 == 0 else h - 1

        # Try multiple codecs for compatibility
        codecs = [
            ("avc1", ".mp4"),
            ("mp4v", ".mp4"),
            ("XVID", ".avi"),
        ]

        writer = None
        for codec, ext in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            if ext != ".mp4":
                output_path = output_path.rsplit(".", 1)[0] + ext
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if writer.isOpened():
                break
            writer.release()
            writer = None

        if writer is None or not writer.isOpened():
            print("[AnimationBuilder] No working video codec found.")
            return AnimationBuilder._fallback_gif(
                frame_paths, os.path.dirname(output_path), fps
            )

        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is not None:
                frame = cv2.resize(frame, (w, h))
                writer.write(frame)

        writer.release()

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        basename = os.path.basename(output_path)
        print(f"[AnimationBuilder] {basename} — {len(frame_paths)} frames, "
              f"{fps} fps, {size_mb:.1f} MB")
        return output_path

    @staticmethod
    def _fallback_gif(frame_paths: list, output_dir: str, fps: int) -> str:
        """Fallback GIF generation if MP4 encoding fails."""
        from PIL import Image

        output_path = os.path.join(output_dir, "trajectory.gif")
        frames = []
        for fp in frame_paths:
            try:
                img = Image.open(fp)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(img)
            except Exception:
                continue

        if len(frames) < 2:
            return ""

        duration_ms = max(100, int(1000 / fps))
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[AnimationBuilder] Fallback GIF — {len(frames)} frames, {size_mb:.1f} MB")
        return output_path
