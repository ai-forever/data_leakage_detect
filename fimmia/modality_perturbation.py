"""
Modality perturbation helpers for generating neighbor variants of image/video/audio inputs.
Used by neighbors.py when modality_column is set and present in the dataset.

Supported modality types are exactly: "image", "video", "audio".
Perturbations are adversarial-style (rotation, distortion, smoothing, silence segments)
rather than optimization-based attacks.
"""

import os
import shutil
from typing import List

import numpy as np

_SUPPORTED_MODALITIES = ("image", "video", "audio")


def _augment_brightness_camera_images(
    image: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """HSV brightness scaling for adversarial-style augmentation."""
    import cv2

    image = image.astype(np.uint8)
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.75 + rng.normal() * 0.5
    image1[:, :, 2] = np.clip(
        image1[:, :, 2].astype(np.float64) * random_bright, 0, 255
    ).astype(np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def _transform_image(
    img: np.ndarray,
    ang_range: float,
    shear_range: float,
    trans_range: float,
    brightness: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply rotation, translation, shear, and optionally brightness.
    Adversarial-style geometric and photometric distortions.
    """
    import cv2

    rows, cols, ch = img.shape
    # Rotation
    ang_rot = rng.uniform(-ang_range / 2, ang_range / 2)
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
    # Translation
    tr_x = trans_range * rng.uniform() - trans_range / 2
    tr_y = trans_range * rng.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + shear_range * rng.uniform() - shear_range / 2
    pt2 = 20 + shear_range * rng.uniform() - shear_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_m = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, rot_m, (cols, rows))
    img = cv2.warpAffine(img, trans_m, (cols, rows))
    img = cv2.warpAffine(img, shear_m, (cols, rows))
    if brightness == 1:
        img = _augment_brightness_camera_images(img, rng)
    return np.clip(img, 0, 255).astype(np.uint8)


def _perturb_image(path: str, output_path: str, seed: int) -> None:
    """Apply adversarial-style perturbations: rotation, translation, shear, brightness; optional negative/contrast."""
    import cv2

    rng = np.random.default_rng(seed)
    img = cv2.imread(path)
    if img is None:
        shutil.copy2(path, output_path)
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _transform_image(
        img, ang_range=5.0, shear_range=3.5, trans_range=5.0, brightness=1, rng=rng
    )
    # Optional negative-like filter for a subset (controlled by seed)
    if (seed % 5) == 2:
        img = 255 - img
    img = np.clip(img, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)


def _perturb_video(path: str, output_path: str, seed: int) -> None:
    """Apply image-like transforms to a subset of frames and drop some frames entirely."""
    import torch
    from torchvision import io

    rng = np.random.default_rng(seed)
    video, audio, info = io.read_video(
        path, start_pts=0.0, end_pts=None, pts_unit="sec", output_format="TCHW"
    )
    T, C, H, W = video.shape
    fps = info["video_fps"]
    if T < 2:
        io.write_video(output_path, video, fps=fps)
        return
    video_np = video.numpy()
    # Drop a subset of frames entirely (e.g. 5–15%)
    drop_ratio = 0.05 + 0.10 * rng.random()
    n_drop = max(0, min(T - 1, int(T * drop_ratio)))
    drop_indices = (
        set(rng.choice(T, size=n_drop, replace=False)) if n_drop > 0 else set()
    )
    keep_indices = [t for t in range(T) if t not in drop_indices]
    if len(keep_indices) < 1:
        keep_indices = [0]
    # Apply image-like transforms (rotation, translation, shear, brightness) to a subset of frames
    transform_ratio = 0.3 + 0.2 * rng.random()
    n_transform = max(1, int(len(keep_indices) * transform_ratio))
    transform_indices = set(
        rng.choice(
            len(keep_indices), size=min(n_transform, len(keep_indices)), replace=False
        )
    )
    for ii in transform_indices:
        t = keep_indices[ii]
        frame = video_np[t].transpose(1, 2, 0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = _transform_image(
            frame,
            ang_range=5.0,
            shear_range=3.5,
            trans_range=5.0,
            brightness=1,
            rng=rng,
        )
        video_np[t] = frame.transpose(2, 0, 1)
    out_frames = video_np[keep_indices]
    video_t = torch.from_numpy(out_frames.astype(np.uint8))
    io.write_video(output_path, video_t, fps=fps)


def _perturb_audio(path: str, output_path: str, seed: int) -> None:
    """Apply random silence segments (simulate removed words); optional volume reduction.

    Audio files are expected to be readable; if not, a clear error is raised.
    """
    rng = np.random.default_rng(seed)

    # Read audio using soundfile with a scipy fallback; raise on failure.
    data = None
    sr = None
    try:
        import soundfile as sf  # type: ignore[import]

        data, sr = sf.read(path, dtype="float64")
    except Exception:
        from scipy.io import wavfile  # type: ignore[import]

        try:
            sr, data = wavfile.read(path)
            if data.dtype in (np.int16, np.int32):
                data = data.astype(np.float64) / np.iinfo(data.dtype).max
        except Exception as e_wav:
            raise RuntimeError(
                f"Failed to read audio file '{path}' with soundfile or scipy.io.wavfile"
            ) from e_wav

    if data is None or sr is None:
        raise RuntimeError(f"Failed to read audio file '{path}'")

    data = np.atleast_1d(data)
    n_samples = data.shape[0]
    duration_s = n_samples / sr
    # 1-3 contiguous silence segments (0.05-0.25 s each) to simulate removed words
    n_segments = rng.integers(1, 4)
    segment_duration_min = 0.05
    segment_duration_max = min(0.25, duration_s / 4)
    if segment_duration_max <= segment_duration_min:
        segment_duration_max = segment_duration_min + 0.01
    for _ in range(n_segments):
        seg_len = int(sr * rng.uniform(segment_duration_min, segment_duration_max))
        seg_len = min(seg_len, n_samples - 1)
        if seg_len <= 0:
            continue
        start = rng.integers(0, max(1, n_samples - seg_len))
        data[start : start + seg_len] = 0.0
    # Optional volume reduction
    if (seed % 4) == 1:
        data = data * 0.7

    # Write audio back; prefer soundfile, with scipy fallback. Raise on failure.
    try:
        import soundfile as sf  # type: ignore[import]

        sf.write(output_path, data, sr)
    except Exception:
        from scipy.io import wavfile  # type: ignore[import]

        try:
            out = np.clip(data * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(output_path, sr, out)
        except Exception as e_wav:
            raise RuntimeError(
                f"Failed to write perturbed audio to '{output_path}'"
            ) from e_wav


def get_modality_neighbors(
    modality_type: str,
    path: str,
    num_neighbors: int,
    output_dir: str,
    row_index: int,
) -> List[str]:
    """
    Generate num_neighbors perturbed versions of a *local* modality asset at path.
    Saves files under output_dir and returns a list of paths (same length as num_neighbors).

    Supported modality_type values are exactly: "image", "video", "audio".
    Any other value returns [path] * num_neighbors with no perturbation.

    The function assumes that `path` is a valid local file path; remote URLs are not supported.
    """
    if path is None or (isinstance(path, float) and np.isnan(path)):
        raise RuntimeError(
            f"Invalid modality path for perturbation (None/NaN) at row {row_index}"
        )
    path = str(path).strip()
    if not path:
        raise RuntimeError(f"Empty modality path for perturbation at row {row_index}")
    if not os.path.isfile(path):
        raise RuntimeError(
            f"Modality path does not exist or is not a file: '{path}' (row {row_index})"
        )
    os.makedirs(output_dir, exist_ok=True)
    base, ext = os.path.splitext(os.path.basename(path))
    if not ext:
        ext = ".bin"
    modality_type = modality_type.lower()
    if modality_type not in _SUPPORTED_MODALITIES:
        raise RuntimeError(f"Unsupported modality type: '{modality_type}'")
    perturb_fn = {
        "image": _perturb_image,
        "video": _perturb_video,
        "audio": _perturb_audio,
    }[modality_type]
    out_paths: List[str] = []
    for k in range(num_neighbors):
        out_name = f"{base}_row{row_index}_n{k}{ext}"
        out_path = os.path.join(output_dir, out_name)
        # Let any errors surface so the user can see what went wrong.
        perturb_fn(path, out_path, seed=hash((path, row_index, k)) % (2**32))
        out_paths.append(out_path)
    return out_paths
