from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lab_utils.visualization import plot_class_balance, plot_numeric_distribution
SEED = 1234
SPLITS = ('train', 'val', 'test')
LABELS = ('cat', 'dog')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def list_image_paths_for_group(data_root: Path, split: str, label: str) -> list[Path]:
    group_dir = data_root / split / label
    paths = []
    if group_dir.exists():
        for file_path in group_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(file_path)
    return paths

def inspect_image_file(path: Path) -> tuple[int, int, float]:
    with Image.open(path) as img:
        img_rgb = img.convert('RGB')
        width, height = img_rgb.size
        img_array = np.array(img_rgb, dtype=np.float32) / 255.0
        mean_intensity = float(img_array.mean())
        return (width, height, mean_intensity)

def make_metadata_row(path: Path, data_root: Path, split: str, label: str) -> dict[str, object]:
    width, height, mean_intensity = inspect_image_file(path)
    return {'filepath': str(path.relative_to(data_root)), 'label': label, 'split': split, 'width': width, 'height': height, 'mean_intensity': mean_intensity}

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            paths = list_image_paths_for_group(data_root, split, label)
            rows.extend((make_metadata_row(p, data_root, split, label) for p in paths))
    return pd.DataFrame(rows).sort_values(['split', 'label', 'filepath']).reset_index(drop=True)

def load_metadata_table(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def summarize_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {'rows': len(frame), 'columns': frame.columns.tolist(), 'class_counts': frame['label'].value_counts(), 'split_counts': frame['split'].value_counts()}

def build_label_split_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(frame['label'], frame['split'])

def audit_metadata(frame: pd.DataFrame) -> dict[str, object]:
    missing_values = frame.isna().sum().to_dict()
    duplicate_filepaths = frame['filepath'].duplicated().sum()
    allowed_labels = {'cat', 'dog'}
    actual_labels = set(frame['label'].unique())
    bad_labels = list(actual_labels - allowed_labels)
    non_positive_sizes = int(((frame['width'] <= 0) | (frame['height'] <= 0)).sum())
    return {'missing_values': missing_values, 'duplicate_filepaths': duplicate_filepaths, 'bad_labels': bad_labels, 'non_positive_sizes': non_positive_sizes}

def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df['pixel_count'] = df['width'] * df['height']
    df['aspect_ratio'] = df['width'] / df['height']
    df['brightness_band'] = pd.qcut(df['mean_intensity'], q=4, labels=['darkest', 'dim', 'bright', 'brightest'])
    ref_pixels = 64 * 64

    def categorize_size(pixels: int) -> str:
        if pixels < ref_pixels:
            return 'small'
        elif pixels == ref_pixels:
            return 'medium'
        else:
            return 'large'
    df['size_bucket'] = df['pixel_count'].apply(categorize_size)
    return df

def build_split_characteristics_table(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.groupby('split')[['width', 'height', 'pixel_count', 'mean_intensity']].mean()
    summary = summary.rename(columns={'width': 'avg_width', 'height': 'avg_height', 'pixel_count': 'avg_pixel_count', 'mean_intensity': 'avg_mean_intensity'})
    return summary

def sample_balanced_by_split_and_label(frame: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    pieces = [
        g.sample(n=min(n_per_group, len(g)), random_state=seed)
        for _, g in frame.groupby(['split', 'label'], sort=True)
    ]
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)
sample_size_per_group = 5
