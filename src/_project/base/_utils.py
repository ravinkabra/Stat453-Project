import os
import re
from typing import Optional, Dict, Any

try:
    # lightweight import; the decorator is no-op on non-plasma installs but available when PL is installed
    from pytorch_lightning.utilities import rank_zero_only
except Exception:  # pragma: no cover - if PL is not available, fallback to a no-op decorator

    def rank_zero_only(fn):
        return fn


def get_next_version(save_dir: str, name: Optional[str] = None, version_prefix: Optional[str] = None) -> str:
    """
    Detects existing version directories (e.g., 'version_0', 'version_1')
    under save_dir/(name if exists) and returns the next available version string.
    If version_prefix is provided (e.g., "1.0."), it tries to find the max 'x' for "1.0.x"
    otherwise, it auto-increments "version_x".

    Args:
        save_dir: The base directory where logs/versions are saved.
        name: Optional, the name of the project, often used as a sub-directory.
        version_prefix: Optional, a version string like "1.0." to find "1.0.x".
                        or "prefix_" to find "prefix_x".
                        If None, it looks for "version_x" directories.
    Returns:
        The next version string (e.g., "1.0.5" or "version_3").
    """
    target_dir = os.path.join(save_dir, name) if name else save_dir
    # Normalize version_prefix so callers may pass either '1.0' or '1.0.'
    # normalized_prefix has no trailing dot; we'll add a single '.' when matching/returning
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)  # Ensure target directory exists for scanning
        if version_prefix:
            return f"{version_prefix}0"
        return "version_0"

    max_version_num = -1
    for item in os.listdir(target_dir):
        if os.path.isdir(os.path.join(target_dir, item)):
            if item == version_prefix:
                return item
            
            if version_prefix:
                # Regex for "1.0.x" or similar. Use version_prefix and require an explicit dot
                match = re.match(rf"^{re.escape(version_prefix)}(\d+)$", item)
                if match:
                    current_x = int(match.group(1))
                    max_version_num = max(max_version_num, current_x)

            else:
                # Regex for "version_x"
                match = re.match(r"^version_(\d+)$", item)
                if match:
                    current_x = int(match.group(1))
                    max_version_num = max(max_version_num, current_x)

    next_version_num = max_version_num + 1
    if version_prefix:
        return f"{version_prefix}{next_version_num}"
    return f"version_{next_version_num}"


@rank_zero_only
def compute_and_set_versions(
    loggers: Optional[Dict[str, Any]],
    save_dir: str,
    name: Optional[str] = None,
    version: Optional[str] = None,
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

    # Delegate version detection to helper which handles directory creation and
    # returns the next version string. project_version takes precedence over
    # version_prefix when provided.
    unified_version = get_next_version(save_dir, name, version_prefix=version)

    # If a logger lacks a version and a project_version exists, set it to the project_version
    for logger_config in loggers.values():
        try:
            if getattr(logger_config, "name", None) is None:
                setattr(logger_config, "name", name)
        except Exception:
            # ignore if attribute setting fails
            pass

    # Apply the unified version to all loggers (best-effort). This keeps all logger
    # versions consistent. When there were no prior versions, unified_version will be
    # exactly the project_version (no numeric suffix); subsequent runs will create
    # project_version.<n>.
    for logger_config in loggers.values():
        try:
            setattr(logger_config, "version", unified_version)
        except Exception:
            pass
    return unified_version
