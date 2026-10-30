from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DATASET = BASE_DIR / "datasets" / "image"
TEXT_DATASET = BASE_DIR / "text_models" / "data" / "ag_news"
VIDEO_DATASET = BASE_DIR / "datasets" / "video"

IMAGE_MODEL = BASE_DIR / "image_models"
TEXT_MODEL = BASE_DIR / "text_models"
VIDEO_MODEL = BASE_DIR / "video_models"