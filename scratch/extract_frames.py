import subprocess
from pathlib import Path

timestamps = [
    ("frame_01_beat_1_1.png", "00:00:01.000"),
    ("frame_02_beat_2_1.png", "00:00:03.500"),
    ("frame_03_beat_2_2.png", "00:00:06.000"),
    ("frame_04_beat_3_1.png", "00:00:10.000"),
    ("frame_05_beat_4_1.png", "00:00:14.500"),
    ("frame_06_beat_4_2.png", "00:00:17.000"),
    ("frame_07_beat_5_1.png", "00:00:18.500"),
    ("frame_08_beat_5_2.png", "00:00:21.000"),
    ("frame_09_beat_6_1.png", "00:00:24.000"),
    ("frame_10_beat_6_2.png", "00:00:26.500"),
    ("frame_11_beat_7_1.png", "00:00:30.000"),
    ("frame_12_beat_8_1.png", "00:00:33.500"),
    ("frame_13_beat_8_2.png", "00:00:36.000"),
]

video_path = "storage/e8342d48-2150-4b46-b05e-88c454f1fc87/final_video_captioned.mp4"
out_dir = Path("scratch/extracted_frames")
out_dir.mkdir(parents=True, exist_ok=True)

for fname, ts in timestamps:
    out_path = out_dir / fname
    cmd = [
        "ffmpeg", "-y",
        "-ss", ts,
        "-i", video_path,
        "-vframes", "1",
        str(out_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"Extracted {fname} at {ts}")
