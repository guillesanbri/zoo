from typing import Dict, Any, Optional

import timm


def get_mixup_fn(
    cfg: Optional[Dict[str, Any]], num_classes: int
) -> Optional[timm.data.Mixup]:
    """
    Instantiates a Mixup object based on the specified configuration.
    """
    if cfg is None:
        return None
    else:
        return timm.data.mixup.Mixup(
            **cfg,
            num_classes=num_classes,
        )
