"""Loader wrappers for lazy frame iteration and optional PyTorch integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional

from . import parser

Transform = Optional[Callable[[Any], Any]]


def iter_frame_samples(
    dataset_root: str,
    transform: Transform = None,
    with_metadata: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield lazy frame samples from discovered records.

    Yields dict with keys:
      - image: loaded frame/image (or None if load fails)
      - frame_idx
      - frame_ref
      - record_id
      - record_type
      - record_path
      - metadata (optional)
    """
    for record_id, record_type, record_path, _ in parser.find_records(dataset_root):
        metadata = parser.record_metadata(record_path, dataset_root=dataset_root)
        for frame_idx, frame_ref in parser.iter_frames(record_path):
            image = parser.get_frame(frame_ref)
            if image is None:
                continue
            if transform is not None:
                image = transform(image)

            sample: Dict[str, Any] = {
                "image": image,
                "frame_idx": frame_idx,
                "frame_ref": frame_ref,
                "record_id": record_id,
                "record_type": record_type,
                "record_path": record_path,
            }
            if with_metadata:
                sample["metadata"] = metadata
            yield sample


class FrameIterable:
    """Simple iterable wrapper useful for torchvision-style pipelines."""

    def __init__(self, dataset_root: str, transform: Transform = None) -> None:
        self.dataset_root = dataset_root
        self.transform = transform

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter_frame_samples(self.dataset_root, transform=self.transform)


try:
    from torch.utils.data import Dataset  # type: ignore

    class TorchFrameDataset(Dataset):
        """PyTorch Dataset wrapper over lazy-discovered frame references.

        Discovery is performed once during initialization (index of frame refs),
        while image loading remains lazy in __getitem__.
        """

        def __init__(self, dataset_root: str, transform: Transform = None) -> None:
            self.dataset_root = dataset_root
            self.transform = transform
            self._entries: List[Dict[str, Any]] = []

            for record_id, record_type, record_path, _ in parser.find_records(dataset_root):
                metadata = parser.record_metadata(record_path, dataset_root=dataset_root)
                for frame_idx, frame_ref in parser.iter_frames(record_path):
                    self._entries.append(
                        {
                            "record_id": record_id,
                            "record_type": record_type,
                            "record_path": record_path,
                            "frame_idx": frame_idx,
                            "frame_ref": frame_ref,
                            "metadata": metadata,
                        }
                    )

        def __len__(self) -> int:
            return len(self._entries)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = dict(self._entries[idx])
            image = parser.get_frame(item["frame_ref"])
            if image is None:
                raise RuntimeError(f"Failed to load frame: {item['frame_ref']}")
            if self.transform is not None:
                image = self.transform(image)
            item["image"] = image
            return item

except Exception:  # pragma: no cover

    class TorchFrameDataset:  # type: ignore
        """Fallback class when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Torch is not installed. Install torch to use TorchFrameDataset.")
