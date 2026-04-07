from __future__ import annotations

import csv
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dearpygui.dearpygui as dpg

from adas.utils.runtime_overrides import get_runtime_overrides_path


@dataclass
class ProcessState:
    proc: Optional[subprocess.Popen[str]] = None
    reader_thread: Optional[threading.Thread] = None
    queue: queue.Queue[Tuple[str, Tuple[int, int, int]]] = field(default_factory=queue.Queue)
    lines: List[Tuple[str, Tuple[int, int, int]]] = field(default_factory=list)
    running: bool = False
    title: str = ""


class MasterDashboard:
    def __init__(self, project_root: str) -> None:
        self.project_root = os.path.abspath(project_root)
        self.index_path = os.path.join(self.project_root, "data/processed/index.db")
        self.dataset_root = os.path.join(self.project_root, "data/raw/DADA2000")
        self.service_name = "adas"
        self.page_size = 200
        self.page_index = 0
        self.sort_column: str = ""
        self.sort_desc: bool = False

        self.rows: List[Dict[str, Any]] = []
        self.filtered_rows: List[Dict[str, Any]] = []
        self.page_rows: List[Dict[str, Any]] = []
        self.columns: List[str] = []
        self.enum_columns: List[str] = []
        self.free_text_columns: List[str] = []
        self.enum_values: Dict[str, List[str]] = {}
        self.enum_map_readable: Dict[Tuple[str, str], str] = {}  # (col, val) -> label
        self.enum_filters: Dict[str, str] = {}
        self.find_filters: Dict[str, Optional[int]] = {"category_id": None, "video_id": None}

        self.selected_row: Optional[Dict[str, Any]] = None
        self.selected_row_id: Optional[str] = None
        self.process = ProcessState()
        self.actions_panel_width = 520

        self._table_tags: List[str] = []
        self._row_highlight_theme: Optional[int] = None
        self._table_card_theme: Optional[int] = None
        self._run_card_theme: Optional[int] = None
        self._has_started_setup = False
        self._process_on_exit: Optional[Callable[[int], None]] = None
        self._active_kill_pattern: Optional[str] = None
        self.startup_log_lines: List[Tuple[str, Tuple[int, int, int]]] = []
        self._startup_queue: queue.Queue[Tuple[str, Tuple[int, int, int]]] = queue.Queue()
        self._docker_setup_running = False
        self._telemetry_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._telemetry_stop = threading.Event()
        self._telemetry_thread: Optional[threading.Thread] = None
        self._latest_telemetry: Dict[str, Any] = {}
        self._prev_cpu_snapshot: Optional[Dict[str, Tuple[int, int]]] = None
        self._param_history: Dict[str, List[str]] = {}
        self._param_tags: List[str] = []
        self._param_undo_suppress = False
        self._hidden_columns: set = {
            "maps_path", "type", "abnormal_start_frame", "total_frames",
            "interval_0_tai", "interval_tai_tco", "interval_tai_tae",
            "interval_tco_tae", "interval_tae_end",
        }
        self.runtime_overrides_path = get_runtime_overrides_path(self.project_root)
        self._runtime_overrides = self._load_runtime_overrides()

    def run(self) -> None:
        dpg.create_context()
        self._build_theme()
        self._build_ui()
        self._apply_runtime_overrides_to_inputs()
        dpg.create_viewport(title="ADAS Control Panel", width=1680, height=980, x_pos=120, y_pos=60, decorated=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("root", True)
        dpg.set_viewport_resize_callback(self._on_viewport_resize)
        self._on_viewport_resize()
        self._reload_table_data(show_message=True)
        self._start_telemetry_thread()

        while dpg.is_dearpygui_running():
            self._drain_startup_output()
            self._drain_process_output()
            self._drain_telemetry_output()
            self._poll_process_end()
            dpg.render_dearpygui_frame()

        self._stop_telemetry_thread()
        self._cancel_active_process()
        dpg.destroy_context()

    def _build_theme(self) -> None:
        with dpg.theme() as row_theme:
            with dpg.theme_component(dpg.mvSelectable):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (70, 120, 200))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (90, 140, 220))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 120, 200))
        self._row_highlight_theme = row_theme

        with dpg.theme() as sel_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (30, 140, 40))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (40, 170, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (30, 140, 40))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
        self._selected_btn_theme = sel_btn_theme

        with dpg.theme() as table_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (220, 70, 70))
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 2)
        self._table_card_theme = table_theme

        with dpg.theme() as run_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 210, 70))
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 2)
        self._run_card_theme = run_theme

    def _build_ui(self) -> None:
        with dpg.window(tag="root", no_title_bar=True, no_move=True, no_resize=True, width=-1, height=-1):
            with dpg.group(horizontal=True):
                self._build_table_card()
                self._build_actions_panel()

        self._build_process_overlay()
        self._build_startup_overlay()
        self._build_log_window()
        self._build_keyboard_handlers()

    def _build_table_card(self) -> None:
        with dpg.child_window(width=-1, height=-1, border=True, tag="table_card"):
            with dpg.group(horizontal=True):
                dpg.add_text("Video Index Table", color=(255, 100, 100))
                dpg.add_spacer(width=14)
                dpg.add_button(label="VIEW LOG", callback=self._open_log_window, width=120, height=24)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Index path:", color=(170, 170, 170))
                dpg.add_text(self.index_path, tag="index_path_text")

            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="find_category", width=120, hint="category_id")
                dpg.add_button(label="FIND", callback=lambda: self._apply_find_filter("category_id"))
                dpg.add_spacer(width=12)
                dpg.add_input_text(tag="find_video", width=120, hint="video_id")
                dpg.add_button(label="FIND", callback=lambda: self._apply_find_filter("video_id"))
                dpg.add_spacer(width=12)
                dpg.add_button(label="Clear Find", callback=self._clear_find_filters)
                dpg.add_spacer(width=12)
                dpg.add_button(label="Reload Index", callback=lambda: self._reload_table_data(show_message=True))

            with dpg.group(horizontal=True):
                dpg.add_text("Sort:")
                dpg.add_combo(items=["(none)"], default_value="(none)", width=260, tag="sort_column_combo", callback=self._on_sort_column_changed)
                dpg.add_button(label="ASC", width=80, tag="sort_order_btn", callback=self._toggle_sort_order)
                dpg.add_spacer(width=12)
                dpg.add_input_text(tag="export_csv_path", width=380, default_value=os.path.join(self.project_root, "data", "processed", "table_export.csv"))
                dpg.add_button(label="Export CSV", callback=self._export_filtered_csv)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Prev", width=80, callback=self._prev_page)
                dpg.add_button(label="Next", width=80, callback=self._next_page)
                dpg.add_spacer(width=10)
                dpg.add_text("page 0/0", tag="page_info", color=(190, 190, 190))

            dpg.add_spacer(height=5)
            with dpg.collapsing_header(label="Column Filters", default_open=False):
                dpg.add_text("Enum-like columns are filterable here.", color=(170, 170, 170))
                dpg.add_group(tag="enum_filter_group")

            dpg.add_spacer(height=5)
            with dpg.child_window(height=-1, border=False, tag="table_wrapper"):
                dpg.add_table(
                    tag="video_table",
                    header_row=True,
                    freeze_rows=1,
                    row_background=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    scrollX=True,
                    scrollY=True,
                    policy=dpg.mvTable_SizingFixedFit,
                )
        if self._table_card_theme is not None:
            dpg.bind_item_theme("table_card", self._table_card_theme)

    def _build_actions_panel(self) -> None:
        with dpg.child_window(
            width=self.actions_panel_width,
            height=-1,
            border=True,
            tag="actions_panel",
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            dpg.add_text("Selected Video", color=(200, 220, 255))
            dpg.add_text("None", tag="selected_video_label", color=(255, 255, 255))
            dpg.add_separator()

            with dpg.child_window(
                height=-1,
                border=False,
                tag="actions_tabs_container",
                no_scrollbar=True,
                no_scroll_with_mouse=True,
            ):
                with dpg.tab_bar():
                    with dpg.tab(label="Simulation"):
                        self._build_run_simulation_card()
                        self._build_play_video_card()
                    with dpg.tab(label="Dataset"):
                        self._build_build_index_card()
                        self._build_sample_conditions_card()
                    with dpg.tab(label="Debug"):
                        self._build_debug_lanes_card()
                        self._build_debug_obstacles_card()
                        self._build_analyze_context_card()
                    with dpg.tab(label="Tests"):
                        self._build_pytest_card()
                    with dpg.tab(label="Context analysis"):
                        self._build_context_analysis_tab()
                    with dpg.tab(label="Parameters"):
                        self._build_parameters_tab()

            dpg.add_spacer(height=6)
            self._build_runtime_metrics_card()

    def _build_run_simulation_card(self) -> None:
        with dpg.child_window(height=265, border=True, tag="card_run_scenario"):
            dpg.add_text("Run Simulation", color=(90, 220, 90))
            dpg.add_text("category_id: null", tag="run_scenario_category")
            dpg.add_text("video_id: null", tag="run_scenario_video")

            with dpg.group(horizontal=True):
                dpg.add_text("ui")
                dpg.add_combo(["dpg", "cv2", "none"], default_value="dpg", width=110, tag="run_scenario_ui")
                dpg.add_text("fps")
                dpg.add_input_text(default_value="30", width=80, tag="run_scenario_fps")
                dpg.add_text("max")
                dpg.add_input_text(default_value="", width=80, hint="frames", tag="run_scenario_max_frames")

            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="audio", default_value=True, tag="run_scenario_audio")
                dpg.add_checkbox(label="dashboard", default_value=True, tag="run_scenario_dashboard")
                dpg.add_checkbox(label="lanes", default_value=True, tag="run_scenario_lanes")
                dpg.add_checkbox(label="obstacles", default_value=True, tag="run_scenario_obstacles")
                dpg.add_checkbox(label="risk", default_value=True, tag="run_scenario_risk")

            dpg.add_button(label="RUN", width=-1, height=42, callback=self._run_scenario_cmd)
        if self._run_card_theme is not None:
            dpg.bind_item_theme("card_run_scenario", self._run_card_theme)

    def _build_build_index_card(self) -> None:
        with dpg.child_window(height=170, border=True):
            dpg.add_text("Build Index", color=(210, 210, 255))
            dpg.add_input_text(label="dataset", default_value="data/raw/DADA2000", width=-1, tag="build_index_dataset")
            dpg.add_input_text(label="index", default_value="data/processed/index.db", width=-1, tag="build_index_path")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="overwrite", default_value=False, tag="build_index_overwrite")
                dpg.add_checkbox(label="check_freshness", default_value=False, tag="build_index_check_freshness")
                dpg.add_checkbox(label="force", default_value=False, tag="build_index_force")
            dpg.add_button(label="RUN build_index.py", width=-1, callback=self._run_build_index_cmd)

    def _build_play_video_card(self) -> None:
        with dpg.child_window(height=130, border=True):
            dpg.add_text("Play Index Video", color=(210, 210, 255))
            with dpg.group(horizontal=True):
                dpg.add_text("delay_ms")
                dpg.add_input_text(default_value="33", width=90, tag="play_video_delay")
                dpg.add_text("max_width")
                dpg.add_input_text(default_value="1280", width=100, tag="play_video_max_width")
                dpg.add_combo(["dpg", "cv2"], default_value="dpg", width=100, tag="play_video_ui")
            dpg.add_button(label="RUN play_index_video.py", width=-1, callback=self._run_play_video_cmd)

    def _build_debug_lanes_card(self) -> None:
        with dpg.child_window(height=150, border=True):
            dpg.add_text("Debug Lanes", color=(210, 210, 255))
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="show_edges", default_value=False, tag="debug_lanes_edges")
                dpg.add_combo(["dpg", "cv2"], default_value="dpg", width=100, tag="debug_lanes_ui")
                dpg.add_text("max_frames")
                dpg.add_input_text(default_value="", width=90, tag="debug_lanes_max_frames")
            with dpg.group(horizontal=True):
                dpg.add_text("delay_ms")
                dpg.add_input_text(default_value="33", width=90, tag="debug_lanes_delay")
                dpg.add_text("max_width")
                dpg.add_input_text(default_value="1280", width=100, tag="debug_lanes_max_width")
            dpg.add_button(label="RUN debug_lanes.py", width=-1, callback=self._run_debug_lanes_cmd)

    def _build_debug_obstacles_card(self) -> None:
        with dpg.child_window(height=150, border=True):
            dpg.add_text("Debug Obstacles", color=(210, 210, 255))
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="with_lanes", default_value=False, tag="debug_obs_with_lanes")
                dpg.add_combo(["dpg", "cv2"], default_value="dpg", width=100, tag="debug_obs_ui")
                dpg.add_text("max_frames")
                dpg.add_input_text(default_value="", width=90, tag="debug_obs_max_frames")
            with dpg.group(horizontal=True):
                dpg.add_text("delay_ms")
                dpg.add_input_text(default_value="33", width=90, tag="debug_obs_delay")
                dpg.add_text("max_width")
                dpg.add_input_text(default_value="1280", width=100, tag="debug_obs_max_width")
            dpg.add_button(label="RUN debug_obstacles.py", width=-1, callback=self._run_debug_obstacles_cmd)

    def _build_analyze_context_card(self) -> None:
        with dpg.child_window(height=140, border=True):
            dpg.add_text("Analyze Context", color=(210, 210, 255))
            with dpg.group(horizontal=True):
                dpg.add_text("every_n")
                dpg.add_input_text(default_value="10", width=90, tag="analyze_every_n")
                dpg.add_checkbox(label="gui", default_value=True, tag="analyze_gui")
                dpg.add_combo(["dpg", "cv2"], default_value="dpg", width=100, tag="analyze_ui")
            with dpg.group(horizontal=True):
                dpg.add_text("max_frames")
                dpg.add_input_text(default_value="", width=100, tag="analyze_max_frames")
                dpg.add_text("delay_ms")
                dpg.add_input_text(default_value="33", width=90, tag="analyze_delay")
            dpg.add_button(label="RUN analyze_video.py", width=-1, callback=self._run_analyze_context_cmd)

    def _build_sample_conditions_card(self) -> None:
        with dpg.child_window(height=165, border=True):
            dpg.add_text("Sample Conditions", color=(210, 210, 255))
            with dpg.group(horizontal=True):
                dpg.add_text("n")
                dpg.add_input_text(default_value="10", width=80, tag="sample_conditions_n")
                dpg.add_text("seed")
                dpg.add_input_text(default_value="42", width=140, tag="sample_conditions_seed")
            with dpg.group(horizontal=True):
                dpg.add_input_text(default_value="data/processed/index.db", width=230, tag="sample_conditions_index")
                dpg.add_input_text(default_value="data/raw/DADA2000", width=230, tag="sample_conditions_root")
            dpg.add_button(label="RUN sample_conditions.py", width=-1, callback=self._run_sample_conditions_cmd)

    def _build_pytest_card(self) -> None:
        with dpg.child_window(height=210, border=True):
            dpg.add_text("Run Tests (pytest in Docker)", color=(230, 200, 120))
            with dpg.group(horizontal=True):
                dpg.add_text("targets", color=(180, 180, 180))
                dpg.add_input_text(default_value="tests", width=255, tag="pytest_targets")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="-q", default_value=True, tag="pytest_q")
                dpg.add_checkbox(label="-x", default_value=False, tag="pytest_x")
                dpg.add_checkbox(label="-vv", default_value=False, tag="pytest_vv")
            with dpg.group(horizontal=True):
                dpg.add_text("extra args", color=(180, 180, 180))
                dpg.add_input_text(default_value="", width=255, tag="pytest_extra")
            dpg.add_button(label="RUN pytest", width=-1, height=40, callback=self._run_pytest_cmd)

    def _build_runtime_metrics_card(self) -> None:
        with dpg.child_window(
            height=245,
            border=True,
            tag="runtime_metrics_card",
            no_scrollbar=True,
            no_scroll_with_mouse=True,
        ):
            dpg.add_text("Docker Runtime Monitor", color=(145, 230, 180))
            dpg.add_text("Refresh: 0.5s", color=(150, 150, 150))
            dpg.add_separator()
            dpg.add_text("Core CPU usage", color=(220, 220, 220))
            for i in range(4):
                dpg.add_text(f"core{i}: n/a", tag=f"docker_core_{i}", color=(180, 180, 180))
            dpg.add_separator()
            dpg.add_text("RAM: n/a", tag="docker_mem", color=(180, 180, 180))
            dpg.add_text("Load avg: n/a", tag="docker_load", color=(180, 180, 180))
            dpg.add_text("Processes: n/a", tag="docker_procs", color=(180, 180, 180))
            dpg.add_text("Container: unknown", tag="docker_status", color=(180, 180, 180))

    def _build_context_analysis_tab(self) -> None:
        """Tab for configuring the background context-analysis thread."""
        with dpg.child_window(height=300, border=True):
            dpg.add_text("Context Analysis Thread", color=(160, 220, 255))
            dpg.add_text(
                "These settings control how often the secondary context-analysis\n"
                "thread runs during scenario playback.",
                color=(170, 170, 170),
                wrap=400,
            )
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text("context_interval (frames):", color=(200, 200, 200))
                dpg.add_input_text(
                    tag="ctx_interval",
                    default_value="5",
                    width=80,
                    hint="5",
                )

            dpg.add_spacer(height=4)
            dpg.add_text(
                "A lower value means more frequent context updates (more\n"
                "accurate mode/weather detection) but higher CPU usage.\n"
                "Recommended: 3-10 frames.",
                color=(150, 150, 150),
                wrap=400,
            )
            dpg.add_separator()
            dpg.add_text("Context analysis in run_scenario.py uses the interval above.", color=(170, 170, 170), wrap=400)
            dpg.add_text("The interval is passed as --context-interval to the script.", color=(170, 170, 170), wrap=400)

    def _build_parameters_tab(self) -> None:
        """Tab for overriding default algorithm parameters."""
        with dpg.child_window(height=-1, border=False, no_scrollbar=True, no_scroll_with_mouse=True):
            dpg.add_text("Algorithm Parameter Overrides", color=(255, 220, 100))
            dpg.add_text(
                "Enter new values and click Save to replace the defaults.\n"
                "Changes take effect on the next run.",
                color=(170, 170, 170),
                wrap=400,
            )
            dpg.add_separator()

            # --- Lane Detection ---
            with dpg.collapsing_header(label="Lane Detection (LaneProcessingConfig)", default_open=True):
                self._param_row("lane_roi_top", "roi_top", "0.38", "Fraction of frame height for ROI top")
                self._param_row("lane_roi_bottom", "roi_bottom", "0.90", "Fraction of frame height for ROI bottom")
                self._param_row("lane_canny_low", "canny_low", "40", "Canny lower threshold")
                self._param_row("lane_canny_high", "canny_high", "120", "Canny upper threshold")
                self._param_row("lane_clahe_clip", "clahe_clip", "1.8", "CLAHE clip limit")
                self._param_row("lane_hough_threshold", "hough_threshold", "35", "HoughLinesP accumulator threshold")
                self._param_row("lane_hough_min_length", "hough_min_length", "30", "HoughLinesP min line length")
                self._param_row("lane_hough_max_gap", "hough_max_gap", "130", "HoughLinesP max line gap")
                self._param_row("lane_damping_alpha", "damping_alpha", "0.65", "Temporal damping blend (0=no update, 1=raw)")
                self._param_row("lane_max_shift_px", "max_shift_px", "80.0", "Max poly shift per frame (px)")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Lane Detection", callback=self._save_lane_params)
                    dpg.add_button(label="Reset to defaults", callback=self._reset_lane_params)

            dpg.add_spacer(height=6)

            # --- Obstacle Detection ---
            with dpg.collapsing_header(label="Obstacle Detection (DetectorConfig)", default_open=True):
                self._param_row("obs_roi_top", "roi_top", "0.30", "Detection ROI top fraction")
                self._param_row("obs_roi_bottom", "roi_bottom", "0.90", "Detection ROI bottom fraction")
                self._param_row("obs_min_area", "min_area", "400.0", "Min contour area (px)")
                self._param_row("obs_mog2_var_threshold", "mog2_var_threshold", "50.0", "MOG2 variance threshold")
                self._param_row("obs_mog2_learning_rate", "mog2_learning_rate", "0.005", "MOG2 learning rate")
                self._param_row("obs_morph_kernel_size", "morph_kernel_size", "5", "Morphological kernel size")
                self._param_row("obs_min_confidence", "min_confidence", "0.3", "Min detection confidence")
                self._param_row("obs_focal_length_px", "focal_length_px", "700.0", "Focal length (px) for distance est.")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Obstacle Detection", callback=self._save_obstacle_params)
                    dpg.add_button(label="Reset to defaults", callback=self._reset_obstacle_params)

            dpg.add_spacer(height=6)

            # --- Collision Risk Estimator ---
            with dpg.collapsing_header(label="Collision Risk (EstimatorConfig)", default_open=True):
                self._param_row("risk_ttc_warning_s", "ttc_warning_s", "3.0", "TTC warning threshold (s)")
                self._param_row("risk_ttc_brake_s", "ttc_brake_s", "1.5", "TTC brake threshold (s)")
                self._param_row("risk_max_ttc_s", "max_ttc_s", "10.0", "Max TTC (s)")
                self._param_row("risk_max_distance_m", "max_distance_m", "40.0", "Max distance for risk (m)")
                self._param_row("risk_proximity_weight", "proximity_weight", "0.35", "Proximity weight in risk score")
                self._param_row("risk_ttc_weight", "ttc_weight", "0.50", "TTC weight in risk score")
                self._param_row("risk_lateral_weight", "lateral_weight", "0.15", "Lateral weight in risk score")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Risk Estimator", callback=self._save_risk_params)
                    dpg.add_button(label="Reset to defaults", callback=self._reset_risk_estimator_params)

            dpg.add_spacer(height=6)

            # --- Decision ---
            with dpg.collapsing_header(label="Collision Risk (DecisionConfig)", default_open=True):
                self._param_row("dec_warn_score_threshold", "warn_score_threshold", "0.35", "WARN score threshold")
                self._param_row("dec_brake_score_threshold", "brake_score_threshold", "0.65", "BRAKE score threshold")
                self._param_row("dec_ttc_warn_s", "ttc_warn_s", "3.0", "TTC WARN threshold (s)")
                self._param_row("dec_ttc_brake_s", "ttc_brake_s", "1.5", "TTC BRAKE threshold (s)")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save Decision Config", callback=self._save_decision_params)
                    dpg.add_button(label="Reset to defaults", callback=self._reset_decision_params)

    def _param_row(self, tag: str, label: str, default: str, hint: str = "") -> None:
        """Helper: one labelled input row in the parameters tab."""
        self._param_tags.append(tag)
        self._param_history[tag] = [str(default)]
        with dpg.group(horizontal=True):
            dpg.add_text(f"{label}:", color=(200, 200, 200))
            dpg.add_input_text(
                tag=tag,
                default_value=default,
                width=220,
                hint=hint,
                callback=self._on_param_input_changed,
                user_data=tag,
            )

    def _on_param_input_changed(self, _sender: Any, value: str, tag: str) -> None:
        if self._param_undo_suppress:
            return
        value_s = str(value)
        hist = self._param_history.setdefault(tag, [])
        if not hist or hist[-1] != value_s:
            hist.append(value_s)
            if len(hist) > 64:
                del hist[0]

    def _build_keyboard_handlers(self) -> None:
        with dpg.handler_registry(tag="global_key_handlers"):
            dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=self._on_ctrl_z)

    def _on_ctrl_z(self, _sender: Any, _app_data: Any) -> None:
        if not (dpg.is_key_down(dpg.mvKey_Control) or dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)):
            return
        active = dpg.get_active_window()
        focused = dpg.get_focused_item()
        target_tag: Optional[str] = None
        if isinstance(focused, int):
            alias = dpg.get_item_alias(focused)
            if alias in self._param_history:
                target_tag = alias
        if target_tag is None and isinstance(active, int):
            # Fallback: undo the most recently edited parameter input.
            for tag in reversed(self._param_tags):
                if self._param_history.get(tag):
                    target_tag = tag
                    break
        if target_tag is None:
            return
        hist = self._param_history.get(target_tag, [])
        if len(hist) < 2:
            return
        hist.pop()
        prev = hist[-1]
        self._param_undo_suppress = True
        try:
            if dpg.does_item_exist(target_tag):
                dpg.set_value(target_tag, prev)
        finally:
            self._param_undo_suppress = False

    # ---- Parameter save callbacks ----

    def _save_lane_params(self) -> None:
        try:
            from adas.lane_detection.processing import LaneProcessingConfig, DEFAULT_PROCESSING_CONFIG
            import adas.lane_detection.processing as _lmod
            kw = {
                "roi_top": float(dpg.get_value("lane_roi_top")),
                "roi_bottom": float(dpg.get_value("lane_roi_bottom")),
                "canny_low": int(float(dpg.get_value("lane_canny_low"))),
                "canny_high": int(float(dpg.get_value("lane_canny_high"))),
                "clahe_clip": float(dpg.get_value("lane_clahe_clip")),
                "hough_threshold": int(float(dpg.get_value("lane_hough_threshold"))),
                "hough_min_length": int(float(dpg.get_value("lane_hough_min_length"))),
                "hough_max_gap": int(float(dpg.get_value("lane_hough_max_gap"))),
                "damping_alpha": float(dpg.get_value("lane_damping_alpha")),
                "max_shift_px": float(dpg.get_value("lane_max_shift_px")),
            }
            # Build a new config that copies remaining defaults
            import dataclasses
            new_cfg = dataclasses.replace(DEFAULT_PROCESSING_CONFIG, **kw)
            _lmod.DEFAULT_PROCESSING_CONFIG = new_cfg
            self._set_override_section("lane", kw)
            self._startup_log("Lane detection defaults saved.", (100, 220, 100))
        except Exception as exc:
            self._startup_log(f"Lane param save error: {exc}", (255, 130, 130))

    def _reset_lane_params(self) -> None:
        defaults = {
            "lane_roi_top": "0.38", "lane_roi_bottom": "0.90",
            "lane_canny_low": "40", "lane_canny_high": "120",
            "lane_clahe_clip": "1.8", "lane_hough_threshold": "35",
            "lane_hough_min_length": "30", "lane_hough_max_gap": "130",
            "lane_damping_alpha": "0.65", "lane_max_shift_px": "80.0",
        }
        for tag, val in defaults.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        try:
            import adas.lane_detection.processing as _lmod
            from adas.lane_detection.processing import LaneProcessingConfig
            _lmod.DEFAULT_PROCESSING_CONFIG = LaneProcessingConfig()
            self._remove_override_section("lane")
            self._startup_log("Lane detection defaults reset.", (200, 200, 200))
        except Exception as exc:
            self._startup_log(f"Lane reset error: {exc}", (255, 130, 130))

    def _save_obstacle_params(self) -> None:
        try:
            from adas.obstacle_detection.detector import DetectorConfig, DEFAULT_DETECTOR_CONFIG
            import adas.obstacle_detection.detector as _dmod
            import dataclasses
            kw = {
                "roi_top": float(dpg.get_value("obs_roi_top")),
                "roi_bottom": float(dpg.get_value("obs_roi_bottom")),
                "min_area": float(dpg.get_value("obs_min_area")),
                "mog2_var_threshold": float(dpg.get_value("obs_mog2_var_threshold")),
                "mog2_learning_rate": float(dpg.get_value("obs_mog2_learning_rate")),
                "morph_kernel_size": int(float(dpg.get_value("obs_morph_kernel_size"))),
                "min_confidence": float(dpg.get_value("obs_min_confidence")),
                "focal_length_px": float(dpg.get_value("obs_focal_length_px")),
            }
            new_cfg = dataclasses.replace(DEFAULT_DETECTOR_CONFIG, **kw)
            _dmod.DEFAULT_DETECTOR_CONFIG = new_cfg
            self._set_override_section("obstacle", kw)
            self._startup_log("Obstacle detection defaults saved.", (100, 220, 100))
        except Exception as exc:
            self._startup_log(f"Obstacle param save error: {exc}", (255, 130, 130))

    def _reset_obstacle_params(self) -> None:
        defaults = {
            "obs_roi_top": "0.30", "obs_roi_bottom": "0.90",
            "obs_min_area": "400.0", "obs_mog2_var_threshold": "50.0",
            "obs_mog2_learning_rate": "0.005", "obs_morph_kernel_size": "5",
            "obs_min_confidence": "0.3", "obs_focal_length_px": "700.0",
        }
        for tag, val in defaults.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        try:
            import adas.obstacle_detection.detector as _dmod
            from adas.obstacle_detection.detector import DetectorConfig
            _dmod.DEFAULT_DETECTOR_CONFIG = DetectorConfig()
            self._remove_override_section("obstacle")
            self._startup_log("Obstacle detection defaults reset.", (200, 200, 200))
        except Exception as exc:
            self._startup_log(f"Obstacle reset error: {exc}", (255, 130, 130))

    def _save_risk_params(self) -> None:
        try:
            from adas.collision_risk.estimator import EstimatorConfig, DEFAULT_ESTIMATOR_CONFIG
            import adas.collision_risk.estimator as _emod
            import dataclasses
            kw = {
                "ttc_warning_s": float(dpg.get_value("risk_ttc_warning_s")),
                "ttc_brake_s": float(dpg.get_value("risk_ttc_brake_s")),
                "max_ttc_s": float(dpg.get_value("risk_max_ttc_s")),
                "max_distance_m": float(dpg.get_value("risk_max_distance_m")),
                "proximity_weight": float(dpg.get_value("risk_proximity_weight")),
                "ttc_weight": float(dpg.get_value("risk_ttc_weight")),
                "lateral_weight": float(dpg.get_value("risk_lateral_weight")),
            }
            new_cfg = dataclasses.replace(DEFAULT_ESTIMATOR_CONFIG, **kw)
            _emod.DEFAULT_ESTIMATOR_CONFIG = new_cfg
            self._set_override_section("estimator", kw)
            self._startup_log("Risk estimator defaults saved.", (100, 220, 100))
        except Exception as exc:
            self._startup_log(f"Risk param save error: {exc}", (255, 130, 130))

    def _reset_risk_estimator_params(self) -> None:
        defaults = {
            "risk_ttc_warning_s": "3.0", "risk_ttc_brake_s": "1.5",
            "risk_max_ttc_s": "10.0", "risk_max_distance_m": "40.0",
            "risk_proximity_weight": "0.35", "risk_ttc_weight": "0.50",
            "risk_lateral_weight": "0.15",
        }
        for tag, val in defaults.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        try:
            import adas.collision_risk.estimator as _emod
            from adas.collision_risk.estimator import EstimatorConfig
            _emod.DEFAULT_ESTIMATOR_CONFIG = EstimatorConfig()
            self._remove_override_section("estimator")
            self._startup_log("Risk estimator defaults reset.", (200, 200, 200))
        except Exception as exc:
            self._startup_log(f"Risk reset error: {exc}", (255, 130, 130))

    def _save_decision_params(self) -> None:
        try:
            from adas.collision_risk.decision import DecisionConfig, DEFAULT_DECISION_CONFIG
            import adas.collision_risk.decision as _decmod
            import dataclasses
            kw = {
                "warn_score_threshold": float(dpg.get_value("dec_warn_score_threshold")),
                "brake_score_threshold": float(dpg.get_value("dec_brake_score_threshold")),
                "ttc_warn_s": float(dpg.get_value("dec_ttc_warn_s")),
                "ttc_brake_s": float(dpg.get_value("dec_ttc_brake_s")),
            }
            new_cfg = dataclasses.replace(DEFAULT_DECISION_CONFIG, **kw)
            _decmod.DEFAULT_DECISION_CONFIG = new_cfg
            self._set_override_section("decision", kw)
            self._startup_log("Decision config defaults saved.", (100, 220, 100))
        except Exception as exc:
            self._startup_log(f"Decision param save error: {exc}", (255, 130, 130))

    def _reset_decision_params(self) -> None:
        defaults = {
            "dec_warn_score_threshold": "0.35", "dec_brake_score_threshold": "0.65",
            "dec_ttc_warn_s": "3.0", "dec_ttc_brake_s": "1.5",
        }
        for tag, val in defaults.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, val)
        try:
            import adas.collision_risk.decision as _decmod
            from adas.collision_risk.decision import DecisionConfig
            _decmod.DEFAULT_DECISION_CONFIG = DecisionConfig()
            self._remove_override_section("decision")
            self._startup_log("Decision config defaults reset.", (200, 200, 200))
        except Exception as exc:
            self._startup_log(f"Decision reset error: {exc}", (255, 130, 130))

    def _build_process_overlay(self) -> None:
        with dpg.window(
            tag="process_overlay",
            modal=True,
            no_title_bar=True,
            no_close=True,
            show=False,
            width=1300,
            height=800,
            pos=(180, 80),
        ):
            dpg.add_text("Process Output", tag="process_overlay_title", color=(255, 255, 255))
            dpg.add_separator()
            with dpg.child_window(tag="process_output_child", height=680, border=True):
                dpg.add_group(tag="process_output_group")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Cancel", width=200, height=36, callback=self._on_cancel_clicked)
                dpg.add_button(label="Copy Log", width=200, height=36, callback=self._copy_process_log)
                dpg.add_button(label="Close", width=200, height=36, callback=self._close_process_overlay)

    def _build_startup_overlay(self) -> None:
        with dpg.window(
            tag="startup_overlay",
            modal=True,
            no_title_bar=True,
            no_close=True,
            show=True,
            width=980,
            height=420,
            pos=(300, 180),
        ):
            dpg.add_text("Startup", color=(255, 255, 255))
            dpg.add_text(
                "Click RUN DOCKER for checks: PulseAudio, X server, docker CLI, docker daemon, compose service, and container start.",
                wrap=920,
                color=(180, 180, 180),
            )
            dpg.add_separator()
            with dpg.child_window(tag="startup_log_child", height=250, border=True):
                dpg.add_group(tag="startup_log_group")
            with dpg.group(horizontal=True):
                dpg.add_button(label="RUN DOCKER", width=220, height=42, callback=lambda s, a, u: self._on_run_docker(check_only=False))
                dpg.add_button(label="Retry Checks", width=220, height=42, callback=lambda s, a, u: self._on_run_docker(check_only=True))
                dpg.add_button(label="Copy Log", width=220, height=42, callback=lambda s, a, u: self._copy_startup_log())
                dpg.add_button(label="Continue", width=220, height=42, callback=lambda s, a, u: dpg.configure_item("startup_overlay", show=False))

    def _build_log_window(self) -> None:
        with dpg.window(
            tag="log_window",
            label="Panel Logs",
            show=False,
            width=980,
            height=520,
            pos=(250, 120),
            no_collapse=True,
        ):
            dpg.add_text("Startup and process logs", color=(220, 220, 220))
            dpg.add_separator()
            with dpg.child_window(tag="log_window_child", height=430, border=True):
                dpg.add_group(tag="log_window_group")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh", width=120, callback=lambda: self._refresh_log_window())
                dpg.add_button(label="Copy", width=120, callback=self._copy_log_window)
                dpg.add_button(label="Close", width=120, callback=lambda: dpg.configure_item("log_window", show=False))

    def _open_log_window(self) -> None:
        self._refresh_log_window()
        dpg.configure_item("log_window", show=True)

    def _refresh_log_window(self) -> None:
        if not dpg.does_item_exist("log_window_group"):
            return
        dpg.delete_item("log_window_group", children_only=True)
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        lines.extend(self.startup_log_lines)
        if self.process.lines:
            if lines:
                lines.append(("", (180, 180, 180)))
            lines.append(("--- process output ---", (160, 200, 255)))
            lines.extend(self.process.lines)
        if not lines:
            lines = [("No logs yet.", (180, 180, 180))]
        for text, color in lines[-3000:]:
            dpg.add_text(text, parent="log_window_group", color=color)
        y_max = dpg.get_y_scroll_max("log_window_child")
        dpg.set_y_scroll("log_window_child", y_max)

    def _copy_log_window(self) -> None:
        merged = [line for line, _ in self.startup_log_lines]
        if self.process.lines:
            merged.append("\n--- process output ---")
            merged.extend([line for line, _ in self.process.lines])
        text = "\n".join(merged)
        if os.name == "nt":
            process = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, text=True)
            process.communicate(input=text)
        else:
            try:
                process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True)
                process.communicate(input=text)
            except FileNotFoundError:
                try:
                    process = subprocess.Popen(["xsel", "-b", "-i"], stdin=subprocess.PIPE, text=True)
                    process.communicate(input=text)
                except FileNotFoundError:
                    pass

    def _startup_log(self, line: str, color: Tuple[int, int, int] = (220, 220, 220)) -> None:
        self._startup_queue.put((line, color))

    def _drain_startup_output(self) -> None:
        drained = 0
        while drained < 200:
            try:
                line, color = self._startup_queue.get_nowait()
            except queue.Empty:
                break
            self.startup_log_lines.append((line, color))
            dpg.add_text(line, parent="startup_log_group", color=color)
            y_max = dpg.get_y_scroll_max("startup_log_child")
            dpg.set_y_scroll("startup_log_child", y_max)
            drained += 1

    def _on_run_docker(self, check_only: bool = False) -> None:
        if self._docker_setup_running:
            self._startup_log("Docker setup already running, please wait...", (255, 200, 120))
            return
        self._docker_setup_running = True
        t = threading.Thread(target=self._run_docker_checks, args=(check_only,), daemon=True)
        t.start()

    def _run_docker_checks(self, check_only: bool = False) -> None:
        try:
            self._run_docker_checks_inner(check_only)
        finally:
            self._docker_setup_running = False

    def _run_docker_checks_inner(self, check_only: bool = False) -> None:
        if not self._has_started_setup:
            self._startup_log("setup started", (100, 220, 100))
            self._has_started_setup = True
        docker_ok = True
        runtime_ok = True

        self._startup_log("[check] docker executable", (160, 210, 255))
        docker_bin = shutil.which("docker")
        if docker_bin:
            self._startup_log(f"docker found: {docker_bin}", (100, 220, 100))
        else:
            docker_ok = False
            self._startup_log("docker not found in PATH", (255, 120, 120))

        if docker_bin:
            self._startup_log("[check] docker daemon", (160, 210, 255))
            rc_info, out_info = self._run_short_command(["docker", "info", "--format", "{{.ServerVersion}}"])
            if rc_info == 0:
                self._startup_log(f"docker daemon OK: {out_info.strip()}", (100, 220, 100))
            else:
                docker_ok = False
                self._startup_log("docker daemon not reachable", (255, 120, 120))

            compose_path = os.path.join(self.project_root, "docker-compose.yml")
            self._startup_log("[check] compose file", (160, 210, 255))
            if os.path.exists(compose_path):
                self._startup_log(f"compose file found: {compose_path}", (100, 220, 100))
            else:
                docker_ok = False
                self._startup_log("docker-compose.yml not found", (255, 120, 120))

            self._startup_log("[check] compose service", (160, 210, 255))
            rc_services, out_services = self._run_short_command(["docker", "compose", "config", "--services"])
            if rc_services == 0 and self.service_name in out_services.split():
                self._startup_log(f"compose service OK: {self.service_name}", (100, 220, 100))
            else:
                docker_ok = False
                self._startup_log(f"compose service '{self.service_name}' missing", (255, 120, 120))

        if os.name == "nt":
            self._startup_log("[check] PulseAudio", (160, 210, 255))
            pa_bat = os.path.join(self.project_root, "scripts", "start_pulseaudio.bat")
            if os.path.exists(pa_bat):
                self._startup_log(f"PulseAudio script found: {pa_bat}", (100, 220, 100))
                self._startup_log("attempting to start: cmd /c start_pulseaudio.bat", (170, 220, 170))
                try:
                    proc = subprocess.Popen(["cmd", "/c", pa_bat], cwd=self.project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(2.0)
                    retcode = proc.poll()
                    if retcode is None:
                        self._startup_log("PulseAudio process started (still running)", (170, 220, 170))
                    elif retcode == 0:
                        self._startup_log("PulseAudio process exited with code 0 (success)", (100, 220, 100))
                    else:
                        _, err = proc.communicate()
                        self._startup_log(f"PulseAudio process exited with code {retcode}: {err[:100]}", (255, 150, 120))
                except Exception as exc:
                    runtime_ok = False
                    self._startup_log(f"PulseAudio start failed: {type(exc).__name__}: {exc}", (255, 120, 120))

                self._startup_log("waiting for PulseAudio TCP port 4713...", (200, 200, 100))
                for attempt in range(5):
                    if self._is_port_open("127.0.0.1", 4713):
                        self._startup_log(f"PulseAudio TCP port 4713 reachable (attempt {attempt+1}/5)", (100, 220, 100))
                        break
                    else:
                        self._startup_log(f"Port check {attempt+1}/5: 127.0.0.1:4713 not reachable", (200, 180, 100))
                        if attempt < 4:
                            time.sleep(0.5)
                else:
                    runtime_ok = False
                    self._startup_log("PulseAudio TCP port 4713 still not reachable after 5 attempts (warning - GUI may not work)", (255, 180, 120))

            else:
                runtime_ok = False
                self._startup_log("PulseAudio script missing (warning - GUI may not work)", (255, 180, 120))
        else:
            self._startup_log("PulseAudio host start skipped (non-Windows runtime)", (220, 220, 140))

        self._startup_log("[check] X server", (160, 210, 255))
        if self._is_x_server_ready_windows():
            self._startup_log("X server check: OK", (100, 220, 100))
        else:
            runtime_ok = False
            self._startup_log("X server check: failed (warning - GUI may not work)", (255, 180, 120))

        if docker_bin and not check_only:
            self._startup_log("[check] docker image", (160, 210, 255))
            rc_img, out_img = self._run_short_command(["docker", "images", "--filter", "reference=adas:latest", "--quiet"])
            if rc_img == 0 and out_img.strip():
                self._startup_log("docker image adas:latest found", (100, 220, 100))
            else:
                self._startup_log("docker image adas:latest not found, building...", (200, 200, 100))
                self._startup_log("running docker compose build (this may take a while)...", (200, 200, 100))
                rc_build, out_build = self._run_short_command(["docker", "compose", "build", self.service_name], timeout=600)
                if rc_build == 0:
                    self._startup_log("docker build completed", (100, 220, 100))
                else:
                    docker_ok = False
                    self._startup_log("docker build failed", (255, 120, 120))
                    for line in out_build.splitlines()[-8:]:
                        self._startup_log(f"  {line}", (255, 160, 160))

            self._startup_log("[check] container status", (160, 210, 255))
            rc_ps_check, out_ps_check = self._run_short_command(["docker", "compose", "ps", "--status", "running", self.service_name])
            if rc_ps_check == 0 and self.service_name in out_ps_check:
                self._startup_log("container already running", (100, 220, 100))
                for line in out_ps_check.splitlines()[-3:]:
                    self._startup_log(f"  {line}", (170, 220, 170))
            else:
                self._startup_log("container not running, starting...", (200, 200, 100))
                self._startup_log(f"running: docker compose up -d {self.service_name}", (170, 200, 170))
                rc, out = self._run_short_command(["docker", "compose", "up", "-d", self.service_name])
                if rc == 0:
                    self._startup_log("docker compose up -d adas -> exit code 0 (OK)", (100, 220, 100))
                    self._startup_log("verifying container status...", (170, 200, 170))
                    rc_ps, out_ps = self._run_short_command(["docker", "compose", "ps", "--status", "running", self.service_name])
                    if rc_ps == 0 and self.service_name in out_ps:
                        for line in out_ps.splitlines()[-3:]:
                            self._startup_log(f"  {line}", (170, 220, 170))
                    else:
                        docker_ok = False
                        self._startup_log("container not reported as running after docker compose up", (255, 120, 120))
                        for line in out_ps.splitlines()[-3:]:
                            self._startup_log("  " + line, (255, 160, 160))
                else:
                    docker_ok = False
                    self._startup_log(f"docker compose up -d adas -> exit code {rc} (FAILED)", (255, 120, 120))
                    self._startup_log("docker compose up error output:", (255, 150, 120))
                    for line in out.splitlines()[-12:]:
                        self._startup_log("  " + line, (255, 160, 160))
        elif check_only:
            self._startup_log("docker compose start skipped (check_only)", (220, 220, 140))
        else:
            self._startup_log("docker executable not found in PATH", (255, 160, 160))

        if docker_ok:
            self._startup_log("startup completed", (100, 220, 100))
        else:
            self._startup_log("CRITICAL: Docker setup incomplete. Fix errors and Retry Checks.", (255, 100, 100))
        
        if not runtime_ok:
            self._startup_log("WARNING: Some runtime features unavailable (GUI/audio may not work fully)", (255, 180, 120))

    def _copy_startup_log(self) -> None:
        text = "\n".join([line for line, _ in self.startup_log_lines])
        if os.name == "nt":
            process = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, text=True)
            process.communicate(input=text)
        else:
            try:
                process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True)
                process.communicate(input=text)
            except FileNotFoundError:
                try:
                    process = subprocess.Popen(["xsel", "-b", "-i"], stdin=subprocess.PIPE, text=True)
                    process.communicate(input=text)
                except FileNotFoundError:
                    pass

    def _is_x_server_ready_windows(self) -> bool:
        for host in ("127.0.0.1", "localhost"):
            if self._is_port_open(host, 6000):
                return True
        return False

    def _on_viewport_resize(self, *_args) -> None:
        """Handle viewport resize - update layout dynamically."""
        try:
            vp_w = dpg.get_viewport_client_width() or 1680
            vp_h = dpg.get_viewport_client_height() or 980
            panel_w = self.actions_panel_width
            table_w = max(800, vp_w - panel_w - 40)
            # Keep a small bottom margin so the root window itself never needs to scroll.
            h = max(500, vp_h - 28)
            if dpg.does_item_exist("table_card"):
                dpg.configure_item("table_card", width=table_w, height=h)
            if dpg.does_item_exist("actions_panel"):
                dpg.configure_item("actions_panel", width=panel_w, height=h)
            telemetry_h = int(h * 0.30)
            telemetry_h = max(200, min(340, telemetry_h))
            if dpg.does_item_exist("runtime_metrics_card"):
                dpg.configure_item("runtime_metrics_card", height=telemetry_h)
            if dpg.does_item_exist("actions_tabs_container"):
                tabs_h = max(180, h - telemetry_h - 96)
                dpg.configure_item("actions_tabs_container", height=tabs_h)
        except Exception:
            pass

    def _start_telemetry_thread(self) -> None:
        if self._telemetry_thread is not None and self._telemetry_thread.is_alive():
            return
        self._telemetry_stop.clear()
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._telemetry_thread.start()

    def _stop_telemetry_thread(self) -> None:
        self._telemetry_stop.set()
        if self._telemetry_thread is not None:
            self._telemetry_thread.join(timeout=1.0)

    def _telemetry_loop(self) -> None:
        while not self._telemetry_stop.is_set():
            stats = self._collect_container_metrics()
            self._telemetry_queue.put(stats)
            self._telemetry_stop.wait(0.5)

    def _collect_container_metrics(self) -> Dict[str, Any]:
        code = (
            "import json; "
            "cpu={}; "
            "f=open('/proc/stat','r'); "
            "lines=f.readlines(); f.close(); "
            "\nfor line in lines:\n"
            "  p=line.split();\n"
            "  if p and p[0].startswith('cpu') and p[0][3:].isdigit():\n"
            "    vals=[int(x) for x in p[1:8]]; cpu[p[0]]=vals;\n"
            "mem={}; "
            "f=open('/proc/meminfo','r'); "
            "\nfor line in f:\n"
            "  if line.startswith('MemTotal:') or line.startswith('MemAvailable:'):\n"
            "    k,v=line.split(':',1); mem[k]=int(v.strip().split()[0]);\n"
            "f.close(); "
            "load=open('/proc/loadavg','r').read().strip().split(); "
            "up=open('/proc/uptime','r').read().strip().split()[0]; "
            "print(json.dumps({'cpu':cpu,'mem':mem,'load':load[:3],'procs':load[3],'uptime':up}))"
        )
        cmd = self._docker_exec_cmd(["python", "-c", code])
        rc, out = self._run_short_command(cmd, timeout=3)
        if rc != 0:
            return {"ok": False, "error": out.strip() or "docker metrics unavailable"}
        try:
            payload = json.loads(out.strip().splitlines()[-1])
        except Exception:
            return {"ok": False, "error": "failed to parse docker metrics"}

        cpu_snap: Dict[str, Tuple[int, int]] = {}
        core_usage: Dict[str, float] = {}
        for core, vals in payload.get("cpu", {}).items():
            if len(vals) < 5:
                continue
            idle = int(vals[3]) + int(vals[4])
            total = sum(int(v) for v in vals)
            cpu_snap[core] = (total, idle)
            if self._prev_cpu_snapshot and core in self._prev_cpu_snapshot:
                prev_total, prev_idle = self._prev_cpu_snapshot[core]
                dt = total - prev_total
                di = idle - prev_idle
                if dt > 0:
                    usage = (1.0 - (di / dt)) * 100.0
                    core_usage[core] = max(0.0, min(100.0, usage))
        self._prev_cpu_snapshot = cpu_snap

        mem = payload.get("mem", {})
        mem_total = float(mem.get("MemTotal", 0.0))
        mem_avail = float(mem.get("MemAvailable", 0.0))
        mem_used = max(0.0, mem_total - mem_avail)
        mem_pct = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0

        return {
            "ok": True,
            "core_usage": core_usage,
            "mem_pct": mem_pct,
            "mem_used_mib": mem_used / 1024.0,
            "mem_total_mib": mem_total / 1024.0,
            "load": payload.get("load", ["n/a", "n/a", "n/a"]),
            "procs": payload.get("procs", "n/a"),
            "uptime_s": float(payload.get("uptime", 0.0) or 0.0),
        }

    def _drain_telemetry_output(self) -> None:
        updated = False
        while True:
            try:
                self._latest_telemetry = self._telemetry_queue.get_nowait()
                updated = True
            except queue.Empty:
                break
        if updated:
            self._render_telemetry()

    def _util_color(self, percent: float) -> Tuple[int, int, int]:
        if percent >= 90.0:
            return (240, 95, 95)
        if percent >= 80.0:
            return (245, 175, 80)
        return (120, 220, 130)

    def _render_telemetry(self) -> None:
        if not self._latest_telemetry:
            return
        stats = self._latest_telemetry
        if not stats.get("ok"):
            msg = stats.get("error", "docker metrics unavailable")
            if dpg.does_item_exist("docker_status"):
                dpg.set_value("docker_status", f"Container: unavailable ({msg[:70]})")
                dpg.configure_item("docker_status", color=(245, 165, 120))
            return

        for i in range(4):
            tag = f"docker_core_{i}"
            if not dpg.does_item_exist(tag):
                continue
            core_name = f"cpu{i}"
            val = stats.get("core_usage", {}).get(core_name)
            if val is None:
                dpg.set_value(tag, f"core{i}: warming up...")
                dpg.configure_item(tag, color=(180, 180, 180))
            else:
                dpg.set_value(tag, f"core{i}: {val:5.1f}%")
                dpg.configure_item(tag, color=self._util_color(val))

        mem_pct = float(stats.get("mem_pct", 0.0))
        mem_used = float(stats.get("mem_used_mib", 0.0))
        mem_total = float(stats.get("mem_total_mib", 0.0))
        dpg.set_value("docker_mem", f"RAM: {mem_used:.0f}/{mem_total:.0f} MiB ({mem_pct:.1f}%)")
        dpg.configure_item("docker_mem", color=self._util_color(mem_pct))

        load = stats.get("load", ["n/a", "n/a", "n/a"])
        dpg.set_value("docker_load", f"Load avg: {load[0]}, {load[1]}, {load[2]}")
        dpg.configure_item("docker_load", color=(180, 210, 255))
        dpg.set_value("docker_procs", f"Processes: {stats.get('procs', 'n/a')}")
        dpg.configure_item("docker_procs", color=(180, 180, 180))
        dpg.set_value("docker_status", "Container: running")
        dpg.configure_item("docker_status", color=(120, 220, 130))

    def _load_runtime_overrides(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.runtime_overrides_path):
                return {}
            with open(self.runtime_overrides_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _persist_runtime_overrides(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.runtime_overrides_path), exist_ok=True)
            with open(self.runtime_overrides_path, "w", encoding="utf-8") as fh:
                json.dump(self._runtime_overrides, fh, indent=2, sort_keys=True)
        except Exception as exc:
            self._startup_log(f"Override save error: {exc}", (255, 130, 130))

    def _set_override_section(self, section: str, values: Dict[str, Any]) -> None:
        self._runtime_overrides[section] = values
        self._persist_runtime_overrides()

    def _remove_override_section(self, section: str) -> None:
        self._runtime_overrides.pop(section, None)
        self._persist_runtime_overrides()

    def _apply_runtime_overrides_to_inputs(self) -> None:
        if not self._runtime_overrides:
            return
        context_cfg = self._runtime_overrides.get("context", {})
        if isinstance(context_cfg, dict) and dpg.does_item_exist("ctx_interval"):
            if "context_interval" in context_cfg:
                dpg.set_value("ctx_interval", str(context_cfg["context_interval"]))

        mapping: List[Tuple[str, str, Dict[str, Any]]] = [
            ("lane", "lane_", self._runtime_overrides.get("lane", {})),
            ("obstacle", "obs_", self._runtime_overrides.get("obstacle", {})),
            ("estimator", "risk_", self._runtime_overrides.get("estimator", {})),
            ("decision", "dec_", self._runtime_overrides.get("decision", {})),
        ]
        for _section, prefix, values in mapping:
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                tag = f"{prefix}{key}"
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, str(value))

    def _is_port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    def _run_short_command(self, cmd: Sequence[str], timeout: int = 15) -> Tuple[int, str]:
        try:
            p = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=timeout)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            return 1, f"Command timed out after {timeout}s"
        except Exception as exc:
            return 1, str(exc)

    def _reload_table_data(self, show_message: bool = False) -> None:
        self.rows, self.columns = self._load_index_rows()
        self._parse_csv_header()
        self._derive_filter_metadata()
        self._build_enum_filter_widgets()
        self._sync_sort_controls()
        self._apply_all_filters()
        if show_message:
            if self.rows:
                self._startup_log(f"index loaded: {len(self.rows)} rows", (120, 220, 120))
            else:
                self._startup_log("index empty or missing. run build_index.py first", (255, 200, 120))

    def _load_index_rows(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not os.path.exists(self.index_path):
            return [], []

        conn = sqlite3.connect(self.index_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(records)")
            records_cols = [r[1] for r in cur.fetchall()]
            cur.execute("PRAGMA table_info(annotations)")
            ann_cols = [r[1] for r in cur.fetchall()]
            ann_cols = [c for c in ann_cols if c not in {"category_id", "video_id"}]

            select_cols = [f"r.{c} AS {c}" for c in records_cols] + [f"a.{c} AS {c}" for c in ann_cols]
            sql = (
                "SELECT "
                + ", ".join(select_cols)
                + " FROM records r LEFT JOIN annotations a "
                + "ON r.category_id = a.category_id AND r.video_id = a.video_id "
                + "ORDER BY r.category_id ASC, r.video_id ASC, r.record_id ASC"
            )
            cur.execute(sql)
            out_rows = [dict(row) for row in cur.fetchall()]
            out_cols = records_cols + ann_cols
            return out_rows, out_cols
        finally:
            conn.close()


    def _parse_csv_header(self) -> None:
        """Parse enum mappings from CSV header."""
        self.enum_map_readable.clear()
        csv_path = os.path.join(self.project_root, "data/raw/DADA2000_video_annotations.csv")
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                header = f.readline().strip().split(';')
            for col_def in header:
                if '(' not in col_def:
                    continue
                col_name = col_def.split('(')[0].strip()
                options_part = col_def.split('(')[1].split(')')[0]
                options = [o.strip() for o in options_part.split(',')]
                for i, label in enumerate(options, start=1):
                    self.enum_map_readable[(col_name, str(i))] = label
        except Exception:
            pass

    def _readable_enum_value(self, col: str, val) -> str:
        """Return readable label for enum value."""
        if val is None:
            return ""
        key = (col, str(val))
        return self.enum_map_readable.get(key, str(val))

    def _derive_filter_metadata(self) -> None:
        self.enum_columns = []
        self.free_text_columns = []
        self.enum_values = {}

        active_filters = dict(self.enum_filters)
        self.enum_filters = {}

        for col in self.columns:
            if col in {"category_id", "video_id"} or col in self._hidden_columns:
                continue

            vals = [r.get(col) for r in self.rows if r.get(col) not in (None, "")]
            if not vals:
                continue

            uniq = sorted({str(v) for v in vals})
            long_text = any(len(str(v)) > 40 for v in vals)
            mostly_text = sum(isinstance(v, str) for v in vals) > (len(vals) * 0.6)

            if len(uniq) <= 20 and not long_text:
                self.enum_columns.append(col)
                self.enum_values[col] = ["All"] + uniq
                self.enum_filters[col] = active_filters.get(col, "All") if active_filters.get(col, "All") in self.enum_values[col] else "All"
            elif mostly_text:
                self.free_text_columns.append(col)

    def _build_enum_filter_widgets(self) -> None:
        dpg.delete_item("enum_filter_group", children_only=True)
        for col in self.enum_columns:
            with dpg.group(horizontal=True, parent="enum_filter_group"):
                dpg.add_text(f"{col}:", color=(180, 180, 180))
                tag = f"filter_{col}"
                # Create readable labels for dropdown
                readable_opts = []
                for opt in self.enum_values[col]:
                    if opt == "All":
                        readable_opts.append(opt)
                    else:
                        readable_opts.append(self._readable_enum_value(col, opt))
                dpg.add_combo(
                    readable_opts,
                    default_value=self.enum_filters.get(col, "All"),
                    width=220,
                    tag=tag,
                    user_data=col,
                    callback=lambda s, a, u: self._on_enum_filter_changed_readable(u, a),
                )

    def _on_enum_filter_changed(self, col: str, value: str) -> None:
        self.enum_filters[col] = value
        self._apply_all_filters()
    
    def _on_enum_filter_changed_readable(self, col: str, readable_value: str) -> None:
        if col is None or col not in self.enum_values:
            return
        if readable_value == "All":
            self.enum_filters[col] = "All"
        else:
            for opt in self.enum_values[col]:
                if self._readable_enum_value(col, opt) == readable_value:
                    self.enum_filters[col] = opt
                    break
        self._apply_all_filters()

    def _apply_find_filter(self, col: str) -> None:
        tag = "find_category" if col == "category_id" else "find_video"
        raw = dpg.get_value(tag).strip()
        self.find_filters[col] = int(raw) if raw.isdigit() else None
        self._apply_all_filters()

    def _clear_find_filters(self) -> None:
        dpg.set_value("find_category", "")
        dpg.set_value("find_video", "")
        self.find_filters = {"category_id": None, "video_id": None}
        self._apply_all_filters()

    def _on_sort_column_changed(self, _sender: Any, value: str) -> None:
        self.sort_column = "" if value == "(none)" else str(value)
        self._apply_all_filters()

    def _toggle_sort_order(self) -> None:
        self.sort_desc = not self.sort_desc
        dpg.set_item_label("sort_order_btn", "DESC" if self.sort_desc else "ASC")
        self._apply_all_filters()

    def _sync_sort_controls(self) -> None:
        sortable_cols = ["(none)"] + [c for c in self.columns if c not in self._hidden_columns]
        dpg.configure_item("sort_column_combo", items=sortable_cols)
        if self.sort_column and self.sort_column in self.columns:
            dpg.set_value("sort_column_combo", self.sort_column)
        else:
            self.sort_column = ""
            dpg.set_value("sort_column_combo", "(none)")
        dpg.set_item_label("sort_order_btn", "DESC" if self.sort_desc else "ASC")

    def _sorted_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.sort_column:
            return rows

        def _key(row: Dict[str, Any]) -> Tuple[int, Any]:
            val = row.get(self.sort_column)
            if val is None:
                return (2, "")
            if isinstance(val, (int, float)):
                return (0, float(val))
            return (1, str(val).lower())

        return sorted(rows, key=_key, reverse=self.sort_desc)

    def _export_filtered_csv(self) -> None:
        if not self.columns:
            self._show_toast("No data available for CSV export.")
            return
        path_raw = dpg.get_value("export_csv_path").strip()
        path = path_raw or os.path.join(self.project_root, "data", "processed", "table_export.csv")
        try:
            path_dir = os.path.dirname(path)
            if path_dir:
                os.makedirs(path_dir, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.columns)
                writer.writeheader()
                for row in self.filtered_rows:
                    writer.writerow({k: row.get(k) for k in self.columns})
            self._show_toast(f"CSV exported: {path} ({len(self.filtered_rows)} rows)")
        except Exception as exc:
            self._show_toast(f"CSV export failed: {exc}")

    def _apply_all_filters(self) -> None:
        out: List[Dict[str, Any]] = []
        for row in self.rows:
            if self.find_filters["category_id"] is not None:
                if row.get("category_id") != self.find_filters["category_id"]:
                    continue
            if self.find_filters["video_id"] is not None:
                if row.get("video_id") != self.find_filters["video_id"]:
                    continue

            blocked = False
            for col, selected in self.enum_filters.items():
                if selected == "All":
                    continue
                if str(row.get(col, "")) != selected:
                    blocked = True
                    break
            if blocked:
                continue
            out.append(row)

        self.filtered_rows = self._sorted_rows(out)
        self.page_index = 0
        self._update_page_rows()
        self._render_table()

    def _update_page_rows(self) -> None:
        total = len(self.filtered_rows)
        page_count = max(1, (total + self.page_size - 1) // self.page_size)
        self.page_index = max(0, min(self.page_index, page_count - 1))
        start = self.page_index * self.page_size
        end = start + self.page_size
        self.page_rows = self.filtered_rows[start:end]
        dpg.set_value("page_info", f"page {self.page_index + 1}/{page_count} - rows {total}")

    def _prev_page(self) -> None:
        if self.page_index > 0:
            self.page_index -= 1
            self._update_page_rows()
            self._render_table()

    def _next_page(self) -> None:
        total = len(self.filtered_rows)
        page_count = max(1, (total + self.page_size - 1) // self.page_size)
        if self.page_index < page_count - 1:
            self.page_index += 1
            self._update_page_rows()
            self._render_table()

    def _format_cell(self, col: str, val) -> str:
        if val is None:
            return ""
        if col == "accident_occurred":
            return "true" if str(val) == "1" else "false"
        return self._readable_enum_value(col, val)

    def _render_table(self) -> None:
        table = "video_table"
        dpg.delete_item(table, children_only=True)
        self._table_tags.clear()

        if not self.columns:
            dpg.add_table_column(label="Status", parent=table)
            with dpg.table_row(parent=table):
                dpg.add_text("index.db not found. run build_index.py first")
            return

        visible_cols = [c for c in self.columns if c not in self._hidden_columns]

        dpg.add_table_column(label="Select", parent=table, init_width_or_weight=80)
        for col in visible_cols:
            dpg.add_table_column(label=col, parent=table)

        for i, row in enumerate(self.page_rows):
            row_tag = f"table_row_{i}"
            self._table_tags.append(row_tag)
            is_selected = self._row_uid(row) == self.selected_row_id
            with dpg.table_row(parent=table, tag=row_tag):
                sel_label = "SELECTED" if is_selected else "Select"
                btn = dpg.add_button(
                    label=sel_label,
                    user_data=row,
                    callback=lambda _s, _a, r: self._on_select_row(r),
                    width=-1,
                    height=22,
                )
                if is_selected and self._selected_btn_theme is not None:
                    dpg.bind_item_theme(btn, self._selected_btn_theme)
                for col in visible_cols:
                    display = self._format_cell(col, row.get(col))
                    if col == "measures":
                        dpg.add_text(display, wrap=300)
                    else:
                        dpg.add_text(display)

            if is_selected and self._row_highlight_theme is not None:
                dpg.bind_item_theme(row_tag, self._row_highlight_theme)

    def _row_uid(self, row: Optional[Dict[str, Any]]) -> str:
        if row is None:
            return "none"
        return f"{row.get('record_id','')}-{row.get('category_id','')}-{row.get('video_id','')}"

    def _on_select_row(self, row: Optional[Dict[str, Any]]) -> None:
        if row is None:
            return
        try:
            self.selected_row = row
            self.selected_row_id = self._row_uid(row)
            cat = row.get("category_id")
            vid = row.get("video_id")
            dpg.set_value("selected_video_label", f"category_id={cat}, video_id={vid}, record_id={row.get('record_id')}")
            dpg.set_value("run_scenario_category", f"category_id: {cat}")
            dpg.set_value("run_scenario_video", f"video_id: {vid}")
        except Exception:
            pass
        self._render_table()

    def _ensure_selected(self) -> Optional[Tuple[int, int]]:
        if not self.selected_row:
            self._show_toast("No selected video. Select one row first.")
            return None
        cat = self.selected_row.get("category_id")
        vid = self.selected_row.get("video_id")
        if not isinstance(cat, int) or not isinstance(vid, int):
            self._show_toast("Selected row has invalid category_id/video_id.")
            return None
        return cat, vid

    def _run_scenario_cmd(self) -> None:
        sel = self._ensure_selected()
        if sel is None:
            return
        cat, vid = sel
        self._set_override_section("context", {"context_interval": dpg.get_value("ctx_interval").strip() or "5"})
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/run_scenario.py",
            "--category-id",
            str(cat),
            "--video-id",
            str(vid),
            "--ui-backend",
            dpg.get_value("run_scenario_ui"),
            "--target-fps",
            dpg.get_value("run_scenario_fps").strip() or "30",
            "--context-interval",
            dpg.get_value("ctx_interval").strip() or "5",
        ])
        max_frames = dpg.get_value("run_scenario_max_frames").strip()
        if max_frames:
            cmd += ["--max-frames", max_frames]

        if not dpg.get_value("run_scenario_audio"):
            cmd.append("--no-audio")
        if not dpg.get_value("run_scenario_dashboard"):
            cmd.append("--no-dashboard")
        if not dpg.get_value("run_scenario_lanes"):
            cmd.append("--no-lanes")
        if not dpg.get_value("run_scenario_obstacles"):
            cmd.append("--no-obstacles")
        if not dpg.get_value("run_scenario_risk"):
            cmd.append("--no-risk")

        self._start_process("Run Simulation", cmd)

    def _run_build_index_cmd(self) -> None:
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/dataset/build_index.py",
            "--dataset-root",
            dpg.get_value("build_index_dataset").strip() or "data/raw/DADA2000",
            "--index-path",
            dpg.get_value("build_index_path").strip() or "data/processed/index.db",
        ])
        if dpg.get_value("build_index_overwrite"):
            cmd.append("--overwrite")
        if dpg.get_value("build_index_check_freshness"):
            cmd.append("--check-freshness")
        if dpg.get_value("build_index_force"):
            cmd.append("--force")
        self._start_process("Build Index", cmd, on_exit=self._reload_after_build)

    def _reload_after_build(self, _code: int) -> None:
        self._reload_table_data(show_message=True)

    def _run_play_video_cmd(self) -> None:
        sel = self._ensure_selected()
        if sel is None:
            return
        cat, vid = sel
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/dataset/play_index_video.py",
            "--index-path",
            "data/processed/index.db",
            "--category-id",
            str(cat),
            "--video-id",
            str(vid),
            "--delay-ms",
            dpg.get_value("play_video_delay").strip() or "33",
            "--max-width",
            dpg.get_value("play_video_max_width").strip() or "1280",
            "--ui-backend",
            dpg.get_value("play_video_ui"),
        ])
        self._start_process("Play Index Video", cmd)

    def _run_debug_lanes_cmd(self) -> None:
        sel = self._ensure_selected()
        if sel is None:
            return
        cat, vid = sel
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/debug_lanes.py",
            "--index-path",
            "data/processed/index.db",
            "--category-id",
            str(cat),
            "--video-id",
            str(vid),
            "--delay-ms",
            dpg.get_value("debug_lanes_delay").strip() or "33",
            "--max-width",
            dpg.get_value("debug_lanes_max_width").strip() or "1280",
            "--ui-backend",
            dpg.get_value("debug_lanes_ui"),
        ])
        max_frames = dpg.get_value("debug_lanes_max_frames").strip()
        if max_frames:
            cmd += ["--max-frames", max_frames]
        if dpg.get_value("debug_lanes_edges"):
            cmd.append("--show-edges")
        self._start_process("Debug Lanes", cmd)

    def _run_debug_obstacles_cmd(self) -> None:
        sel = self._ensure_selected()
        if sel is None:
            return
        cat, vid = sel
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/debug_obstacles.py",
            "--index-path",
            "data/processed/index.db",
            "--category-id",
            str(cat),
            "--video-id",
            str(vid),
            "--delay-ms",
            dpg.get_value("debug_obs_delay").strip() or "33",
            "--max-width",
            dpg.get_value("debug_obs_max_width").strip() or "1280",
            "--ui-backend",
            dpg.get_value("debug_obs_ui"),
        ])
        max_frames = dpg.get_value("debug_obs_max_frames").strip()
        if max_frames:
            cmd += ["--max-frames", max_frames]
        if dpg.get_value("debug_obs_with_lanes"):
            cmd.append("--with-lanes")
        self._start_process("Debug Obstacles", cmd)

    def _run_analyze_context_cmd(self) -> None:
        sel = self._ensure_selected()
        if sel is None:
            return
        cat, vid = sel
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/context/analyze_video.py",
            "--index-path",
            "data/processed/index.db",
            "--category-id",
            str(cat),
            "--video-id",
            str(vid),
            "--every-n",
            dpg.get_value("analyze_every_n").strip() or "10",
            "--delay-ms",
            dpg.get_value("analyze_delay").strip() or "33",
            "--ui-backend",
            dpg.get_value("analyze_ui"),
        ])
        max_frames = dpg.get_value("analyze_max_frames").strip()
        if max_frames:
            cmd += ["--max-frames", max_frames]
        if dpg.get_value("analyze_gui"):
            cmd.append("--gui")
        self._start_process("Analyze Context", cmd)

    def _run_sample_conditions_cmd(self) -> None:
        cmd = self._docker_exec_cmd([
            "python",
            "scripts/dataset/sample_conditions.py",
            "--n",
            dpg.get_value("sample_conditions_n").strip() or "10",
            "--seed",
            dpg.get_value("sample_conditions_seed").strip() or "42",
            "--index-path",
            dpg.get_value("sample_conditions_index").strip() or "data/processed/index.db",
            "--dataset-root",
            dpg.get_value("sample_conditions_root").strip() or "data/raw/DADA2000",
        ])
        self._start_process("Sample Conditions", cmd)

    def _run_pytest_cmd(self) -> None:
        targets = dpg.get_value("pytest_targets").strip() or "tests"
        cmd = self._docker_exec_cmd(["pytest"]) + targets.split()
        if dpg.get_value("pytest_q"):
            cmd.append("-q")
        if dpg.get_value("pytest_x"):
            cmd.append("-x")
        if dpg.get_value("pytest_vv"):
            cmd.append("-vv")
        extra = dpg.get_value("pytest_extra").strip()
        if extra:
            cmd.extend(shlex.split(extra))
        self._start_process("Pytest", cmd)

    def _docker_exec_cmd(self, tail_cmd: List[str]) -> List[str]:
        return ["docker", "compose", "exec", "-T", "-e", "PYTHONUNBUFFERED=1", self.service_name] + tail_cmd

    def _start_process(self, title: str, cmd: List[str], on_exit: Optional[Any] = None) -> None:
        if self.process.running:
            self._show_toast("A process is already running. Cancel it first.")
            return

        dpg.configure_item("process_overlay", show=True)
        dpg.set_value("process_overlay_title", f"{title} - live output")
        dpg.delete_item("process_output_group", children_only=True)
        self.process.lines.clear()
        self.process.title = title

        color = (140, 200, 255)
        self._append_process_line("$ " + " ".join(shlex.quote(p) for p in cmd), color)
        self._active_kill_pattern = self._extract_kill_pattern(cmd)

        kwargs: Dict[str, Any] = dict(
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        if os.name != "nt":
            kwargs["preexec_fn"] = os.setsid
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except Exception as exc:
            self._append_process_line(f"[spawn error] {exc}", (255, 130, 130))
            self.process.running = False
            return

        self.process.proc = proc
        self.process.running = True

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                text, col = self._ansi_to_colored_text(line.rstrip("\n"))
                self.process.queue.put((text, col))
            proc.stdout.close()

        t = threading.Thread(target=_reader, daemon=True)
        self.process.reader_thread = t
        t.start()
        self._process_on_exit = on_exit

    def _extract_kill_pattern(self, cmd: List[str]) -> Optional[str]:
        if "pytest" in cmd:
            return "pytest"
        try:
            py_idx = cmd.index("python")
            if py_idx + 1 < len(cmd):
                script = cmd[py_idx + 1]
                if script.startswith("scripts/"):
                    return script
        except ValueError:
            return None
        return None

    def _ansi_to_colored_text(self, line: str) -> Tuple[str, Tuple[int, int, int]]:
        color = (220, 220, 220)
        if "\x1b[31m" in line:
            color = (255, 120, 120)
        elif "\x1b[32m" in line:
            color = (120, 255, 120)
        elif "\x1b[33m" in line:
            color = (255, 230, 120)
        elif "\x1b[34m" in line:
            color = (120, 170, 255)
        elif "\x1b[35m" in line:
            color = (240, 140, 255)
        elif "\x1b[36m" in line:
            color = (140, 230, 255)

        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        return clean, color

    def _append_process_line(self, line: str, color: Tuple[int, int, int]) -> None:
        self.process.lines.append((line, color))
        if len(self.process.lines) > 4000:
            self.process.lines = self.process.lines[-3000:]
            dpg.delete_item("process_output_group", children_only=True)
            for txt, col in self.process.lines:
                dpg.add_text(txt, parent="process_output_group", color=col)
        else:
            dpg.add_text(line, parent="process_output_group", color=color)

        y_max = dpg.get_y_scroll_max("process_output_child")
        dpg.set_y_scroll("process_output_child", y_max)

    def _drain_process_output(self) -> None:
        drained = 0
        while drained < 300:
            try:
                line, col = self.process.queue.get_nowait()
            except queue.Empty:
                break
            self._append_process_line(line, col)
            drained += 1

    def _poll_process_end(self) -> None:
        if not self.process.running or self.process.proc is None:
            return
        code = self.process.proc.poll()
        if code is None:
            return

        # Drain any remaining output from the reader thread
        if self.process.reader_thread is not None:
            self.process.reader_thread.join(timeout=2.0)
        while True:
            try:
                line, col = self.process.queue.get_nowait()
                self._append_process_line(line, col)
            except queue.Empty:
                break

        self.process.running = False
        status_col = (120, 255, 120) if code == 0 else (255, 130, 130)
        self._append_process_line(f"[exit] return code: {code}", status_col)
        if self._process_on_exit is not None:
            try:
                self._process_on_exit(code)
            finally:
                self._process_on_exit = None

    def _on_cancel_clicked(self) -> None:
        self._cancel_active_process()

    def _cancel_active_process(self) -> None:
        if not self.process.running or self.process.proc is None:
            return

        proc = self.process.proc
        self._append_process_line("[cancel] terminating process...", (255, 210, 120))
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception as exc:
            self._append_process_line(f"[cancel] failed: {exc}", (255, 130, 130))
            return

        deadline = time.time() + 2.0
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.05)

        if proc.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass

        if self._active_kill_pattern:
            self._run_short_command(
                [
                    "docker",
                    "compose",
                    "exec",
                    "-T",
                    self.service_name,
                    "pkill",
                    "-f",
                    self._active_kill_pattern,
                ]
            )

        self.process.running = False
        self._active_kill_pattern = None

    def _close_process_overlay(self) -> None:
        if self.process.running:
            self._show_toast("Process is still running. Cancel first.")
            return
        dpg.configure_item("process_overlay", show=False)

    def _copy_process_log(self) -> None:
        text = "\n".join([line for line, _ in self.process.lines])
        if os.name == "nt":
            process = subprocess.Popen(["clip.exe"], stdin=subprocess.PIPE, text=True)
            process.communicate(input=text)
        else:
            try:
                process = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True)
                process.communicate(input=text)
            except FileNotFoundError:
                try:
                    process = subprocess.Popen(["xsel", "-b", "-i"], stdin=subprocess.PIPE, text=True)
                    process.communicate(input=text)
                except FileNotFoundError:
                    pass

    def _show_toast(self, text: str) -> None:
        dpg.configure_item("startup_overlay", show=True)
        self._startup_log(text, (255, 200, 120))


def run_master_dashboard(project_root: str) -> None:
    app = MasterDashboard(project_root)
    app.run()
