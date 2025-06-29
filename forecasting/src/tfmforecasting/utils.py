import fnmatch, os
from pathlib import Path
from typing import Optional


def get_housing_unit_name(file: Path, separator: str = '_') -> str:
    file_name = file.stem
    file_name_parts = file_name.split(separator)
    return file_name_parts[0] if len(file_name_parts) > 0 else ''


def find_cluster_files(
        data_dir: Path, cluster_id: int, year: Optional[int] = None, month: Optional[int] = None
) -> list[Path]:
    pattern_parts = []
    if year:
        pattern_parts.append(f"{year:04d}")
    if month:
        pattern_parts.append(f"{month:02d}")
    pattern_parts.append(f"_id_{cluster_id}.csv")
    pattern = '*' + '*'.join(pattern_parts)
    result = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(Path(os.path.join(root, name)))
    return result
