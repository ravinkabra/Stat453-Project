import os
import re
from typing import Optional, Dict, Any

try:
    # lightweight import; the decorator is no-op on non-plasma installs but available when PL is installed
    from pytorch_lightning.utilities import rank_zero_only
except Exception:  # pragma: no cover - if PL is not available, fallback to a no-op decorator

    def rank_zero_only(fn):
        return fn


@rank_zero_only
def compute_and_set_versions(
    loggers: Optional[Dict[str, Any]],
    base_save_dir: str,
    project_name: Optional[str] = None,
    version_prefix: Optional[str] = None,
):
    """Compute a next-version string (version_x or prefix.x) and set it on logger configs.

    This function is intended to be run only on rank0 (decorated with `rank_zero_only`).
    It will inspect `base_save_dir` / `project_name` for existing version directories and
    compute the next available numeric suffix. It then assigns the resulting version
    to each logger config that supports a `version` attribute and ensures `name` is set.

    Returns the chosen version string, or None on failure / if `loggers` is falsy.
    """
    if not loggers:
        return None

    target_dir = os.path.join(base_save_dir, project_name) if project_name else base_save_dir
    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception:
        # proceed even if mkdir fails; we'll still try to scan if possible
        pass

    max_version_num = -1
    try:
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            if not os.path.isdir(item_path):
                continue
            if version_prefix:
                m = re.match(rf"^{re.escape(version_prefix)}\.(\d+)$", item)
                if m:
                    current = int(m.group(1))
                    if current > max_version_num:
                        max_version_num = current
            else:
                m = re.match(r"^version_(\d+)$", item)
                if m:
                    current = int(m.group(1))
                    if current > max_version_num:
                        max_version_num = current
    except FileNotFoundError:
        # target_dir missing or inaccessible
        max_version_num = -1

    next_num = max_version_num + 1
    unified_version = f"{version_prefix}.{next_num}" if version_prefix else f"version_{next_num}"

    # Apply to logger configs where possible
    for logger_config in loggers.values():
        try:
            if getattr(logger_config, "name", None) is None:
                setattr(logger_config, "name", project_name)
        except Exception:
            # ignore if attribute setting fails
            pass
        try:
            setattr(logger_config, "version", unified_version)
        except Exception:
            pass

    return unified_version
