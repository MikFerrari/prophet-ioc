"""Optional experiment tracking with Weights & Biases (wandb) for train.py and eval.py.

`Tracker(cfg.wandb, job_type, name, config)` starts a wandb run when `wandb.enabled` is true; every method is a no-op
otherwise, or when wandb is not installed. Online mode needs a wandb login (`wandb login`, or WANDB_API_KEY): without
one the run falls back to offline mode instead of stopping at an interactive prompt (a training started in a detached
screen session would otherwise hang); `wandb sync output/wandb/offline-run-*` uploads it later. Runs are written under
output/wandb/ (git-ignored).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


class Tracker:
    def __init__(self, wandb_cfg: Any, job_type: str, name: str, config: Optional[Dict] = None,
                 directory: str = "output"):
        self.run = None
        if wandb_cfg is None or not wandb_cfg.get("enabled", False):
            return
        try:
            import wandb
        except ImportError:
            print("wandb is not installed (pip install wandb): tracking disabled")
            return
        mode = str(wandb_cfg.get("mode", "online"))
        if mode == "online" and not (os.environ.get("WANDB_API_KEY") or _has_login(wandb)):
            print("wandb: no login found (run `wandb login`): logging offline, upload later with `wandb sync`")
            mode = "offline"
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(project=wandb_cfg.get("project", "prophet-ioc"), entity=wandb_cfg.get("entity"),
                              group=wandb_cfg.get("group"), tags=list(wandb_cfg.get("tags") or []), job_type=job_type,
                              name=name, config=config, mode=mode, dir=directory)
        self._wandb = wandb
        print(f"wandb: {mode} run '{name}'" + (f", {self.run.url}" if mode == "online" and self.run.url else ""))

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.run is not None:
            self.run.log(data, step=step)

    def summary(self, data: Dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(data)

    def images(self, paths, prefix: str = "figures") -> None:
        """Logs image files (paths that do not exist or are not images are skipped)."""
        if self.run is None:
            return
        data = {f"{prefix}/{Path(p).stem}": self._wandb.Image(str(p)) for p in paths
                if p is not None and Path(p).suffix.lower() in (".png", ".jpg", ".jpeg") and Path(p).exists()}
        if data:
            self.run.log(data)

    def table(self, key: str, columns, rows) -> None:
        if self.run is not None:
            self.run.log({key: self._wandb.Table(columns=list(columns), data=[list(r) for r in rows])})

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


def _has_login(wandb) -> bool:
    try:
        return bool(wandb.Api().api_key)
    except Exception:
        return False
