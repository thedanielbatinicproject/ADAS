"""Tests for key mapping in adas.ui.backend_cv2."""

from __future__ import annotations

from adas.ui.backend_cv2 import _map_key
from adas.ui.types import UICommand


class TestMapKey:
    def test_none_keys(self):
        assert _map_key(-1) == UICommand.NONE
        assert _map_key(255) == UICommand.NONE

    def test_quit_keys(self):
        assert _map_key(ord("q")) == UICommand.QUIT
        assert _map_key(27) == UICommand.QUIT

    def test_play_pause_keys(self):
        assert _map_key(ord(" ")) == UICommand.PLAY
        assert _map_key(ord("p")) == UICommand.PLAY
        assert _map_key(ord("P")) == UICommand.PLAY

    def test_step_forward_keys(self):
        assert _map_key(ord("d")) == UICommand.STEP_FWD
        assert _map_key(ord("D")) == UICommand.STEP_FWD
        assert _map_key(83) == UICommand.STEP_FWD
        assert _map_key(2555904) == UICommand.STEP_FWD
        assert _map_key(65363) == UICommand.STEP_FWD

    def test_step_back_keys(self):
        assert _map_key(ord("a")) == UICommand.STEP_BACK
        assert _map_key(ord("A")) == UICommand.STEP_BACK
        assert _map_key(81) == UICommand.STEP_BACK
        assert _map_key(2424832) == UICommand.STEP_BACK
        assert _map_key(65361) == UICommand.STEP_BACK
