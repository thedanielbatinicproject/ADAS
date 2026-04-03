"""Shared fixtures for context tests - synthetic frames."""

import numpy as np
import pytest


@pytest.fixture
def black_frame():
    """Simulate night / very dark scene."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """Simulate overexposed / glare scene."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def gray_frame():
    """Uniform mid-gray frame."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def gradient_frame():
    """Horizontal gradient (dark -> bright) simulating moderate visibility."""
    h, w = 480, 640
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return np.stack([grad, grad, grad], axis=-1)


@pytest.fixture
def noisy_frame():
    """Random-noise frame simulating degraded conditions."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def edges_frame():
    """Frame with sharp vertical edges (alternating black/white strips)."""
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for col_start in range(0, w, 20):
        frame[:, col_start : col_start + 10, :] = 255
    return frame
