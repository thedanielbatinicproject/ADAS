from __future__ import annotations

import csv
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
        self.enum_filters: Dict[str, str] = {}
        self.find_filters: Dict[str, Optional[int]] = {"category_id": None, "video_id": None}

        self.selected_row: Optional[Dict[str, Any]] = None
        self.selected_row_id: Optional[str] = None
        self.process = ProcessState()

        self._table_tags: List[str] = []
        self._row_highlight_theme: Optional[int] = None
        self._table_card_theme: Optional[int] = None
        self._run_card_theme: Optional[int] = None
        self._has_started_setup = False
        self._process_on_exit: Optional[Callable[[int], None]] = None
        self._active_kill_pattern: Optional[str] = None
        self.startup_log_lines: List[Tuple[str, Tuple[int, int, int]]] = []

    def run(self) -> None:
        dpg.create_context()
        self._build_theme()
        self._build_ui()
        dpg.create_viewport(title="ADAS Control Panel", width=1680, height=980, x_pos=120, y_pos=60, decorated=True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._reload_table_data(show_message=True)

        while dpg.is_dearpygui_running():
            self._drain_process_output()
            self._poll_process_end()
            dpg.render_dearpygui_frame()

        self._cancel_active_process()
        dpg.destroy_context()

    def _build_theme(self) -> None:
        with dpg.theme() as row_theme:
            with dpg.theme_component(dpg.mvSelectable):
                dpg.add_theme_color(dpg.mvThemeCol_Header, (70, 120, 200))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (90, 140, 220))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 120, 200))
        self._row_highlight_theme = row_theme

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

    def _build_table_card(self) -> None:
        with dpg.child_window(width=1180, height=-1, border=True, tag="table_card"):
            dpg.add_text("Video Index Table", color=(255, 100, 100))
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
                dpg.add_text("Enum-like columns are filterable here. Free text columns are display-only.", color=(170, 170, 170))
                with dpg.child_window(height=100, border=False):
                    dpg.add_group(tag="enum_filter_group")

            dpg.add_spacer(height=5)
            with dpg.child_window(height=400, border=False):
                dpg.add_table(
                    tag="video_table",
                    header_row=True,
                    row_background=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                    scrollX=True,
                    scrollY=True,
                    policy=dpg.mvTable_SizingStretchProp,
                )
        if self._table_card_theme is not None:
            dpg.bind_item_theme("table_card", self._table_card_theme)

    def _build_actions_panel(self) -> None:
        with dpg.child_window(width=-1, height=-1, border=True):
            dpg.add_text("Selected Video (OV)", color=(200, 220, 255))
            dpg.add_text("None", tag="selected_video_label", color=(255, 255, 255))
            dpg.add_separator()

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
            dpg.add_input_text(label="targets", default_value="tests", width=-1, tag="pytest_targets")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="-q", default_value=True, tag="pytest_q")
                dpg.add_checkbox(label="-x", default_value=False, tag="pytest_x")
                dpg.add_checkbox(label="-vv", default_value=False, tag="pytest_vv")
            dpg.add_input_text(label="extra args", default_value="", width=-1, tag="pytest_extra")
            dpg.add_button(label="RUN pytest", width=-1, height=40, callback=self._run_pytest_cmd)

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
                dpg.add_button(label="RUN DOCKER", width=220, height=42, callback=self._on_run_docker)
                dpg.add_button(label="Retry Checks", width=220, height=42, callback=lambda: self._on_run_docker(check_only=True))
                dpg.add_button(label="Copy Log", width=220, height=42, callback=self._copy_startup_log)
                dpg.add_button(label="Continue", width=220, height=42, callback=lambda: dpg.configure_item("startup_overlay", show=False))

    def _startup_log(self, line: str, color: Tuple[int, int, int] = (220, 220, 220)) -> None:
        self.startup_log_lines.append((line, color))
        dpg.add_text(line, parent="startup_log_group", color=color)
        y_max = dpg.get_y_scroll_max("startup_log_child")
        dpg.set_y_scroll("startup_log_child", y_max)

    def _on_run_docker(self, check_only: bool = False) -> None:
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
                if not check_only:
                    self._startup_log("starting PulseAudio...", (170, 220, 170))
                    try:
                        subprocess.Popen(["cmd", "/c", pa_bat], cwd=self.project_root)
                        time.sleep(1.0)
                    except Exception as exc:
                        runtime_ok = False
                        self._startup_log(f"PulseAudio start failed: {exc}", (255, 120, 120))

                if self._is_port_open("127.0.0.1", 4713):
                    self._startup_log("PulseAudio TCP port 4713 reachable", (100, 220, 100))
                else:
                    runtime_ok = False
                    self._startup_log("PulseAudio TCP port 4713 not reachable (warning - GUI may not work)", (255, 180, 120))
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
            rc_img, out_img = self._run_short_command(["docker", "images", "--filter", f"reference=adas:latest", "--quiet"])
            if rc_img == 0 and out_img.strip():
                self._startup_log("docker image adas:latest found", (100, 220, 100))
            else:
                self._startup_log("docker image adas:latest not found, building...", (200, 200, 100))
                self._startup_log("running docker compose build (this may take a while)...", (200, 200, 100))
                rc_build, out_build = self._run_short_command(["docker", "compose", "build", self.service_name])
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
                rc, out = self._run_short_command(["docker", "compose", "up", "-d", self.service_name])
                if rc == 0:
                    self._startup_log("docker compose up -d adas -> OK", (100, 220, 100))
                    rc_ps, out_ps = self._run_short_command(["docker", "compose", "ps", "--status", "running", self.service_name])
                    if rc_ps == 0 and self.service_name in out_ps:
                        for line in out_ps.splitlines()[-3:]:
                            self._startup_log(f"  {line}", (170, 220, 170))
                    else:
                        docker_ok = False
                        self._startup_log("container not reported as running", (255, 120, 120))
                else:
                    docker_ok = False
                    self._startup_log("docker compose up -d adas -> failed", (255, 120, 120))
                    for line in out.splitlines()[-8:]:
                        self._startup_log(f"  {line}", (255, 160, 160))
        elif check_only:
            self._startup_log("docker compose start skipped (check_only)", (220, 220, 140))
        else:
            self._startup_log("docker executable not found in PATH", (255, 160, 160))

        self._reload_table_data(show_message=False)

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

    def _is_port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    def _run_short_command(self, cmd: Sequence[str]) -> Tuple[int, str]:
        try:
            p = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=90)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except Exception as exc:
            return 1, str(exc)

    def _reload_table_data(self, show_message: bool = False) -> None:
        self.rows, self.columns = self._load_index_rows()
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

    def _derive_filter_metadata(self) -> None:
        self.enum_columns = []
        self.free_text_columns = []
        self.enum_values = {}

        active_filters = dict(self.enum_filters)
        self.enum_filters = {}

        for col in self.columns:
            if col in {"category_id", "video_id"}:
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
                dpg.add_combo(
                    self.enum_values[col],
                    default_value=self.enum_filters.get(col, "All"),
                    width=220,
                    tag=tag,
                    callback=lambda _s, val, c=col: self._on_enum_filter_changed(c, val),
                )

    def _on_enum_filter_changed(self, col: str, value: str) -> None:
        self.enum_filters[col] = value
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
        sortable_cols = ["(none)"] + list(self.columns)
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

    def _render_table(self) -> None:
        table = "video_table"
        dpg.delete_item(table, children_only=True)
        self._table_tags.clear()

        if not self.columns:
            dpg.add_table_column(label="Status", parent=table)
            with dpg.table_row(parent=table):
                dpg.add_text("index.db not found. run build_index.py first")
            return

        dpg.add_table_column(label="Select", parent=table, init_width_or_weight=0.07)
        for col in self.columns:
            dpg.add_table_column(label=col, parent=table)

        for i, row in enumerate(self.page_rows):
            row_tag = f"table_row_{i}"
            self._table_tags.append(row_tag)
            with dpg.table_row(parent=table, tag=row_tag):
                sel_label = "Selected" if self._row_uid(row) == self.selected_row_id else "Select"
                dpg.add_button(
                    label=sel_label,
                    callback=lambda _s, _a, r=row: self._on_select_row(r),
                    width=-1,
                    height=22,
                )
                for col in self.columns:
                    val = row.get(col)
                    dpg.add_text("" if val is None else str(val))

            if self._row_uid(row) == self.selected_row_id and self._row_highlight_theme is not None:
                dpg.bind_item_theme(row_tag, self._row_highlight_theme)

    def _row_uid(self, row: Dict[str, Any]) -> str:
        return f"{row.get('record_id','')}-{row.get('category_id','')}-{row.get('video_id','')}"

    def _on_select_row(self, row: Dict[str, Any]) -> None:
        self.selected_row = row
        self.selected_row_id = self._row_uid(row)
        cat = row.get("category_id")
        vid = row.get("video_id")
        dpg.set_value("selected_video_label", f"category_id={cat}, video_id={vid}, record_id={row.get('record_id')}")
        dpg.set_value("run_scenario_category", f"category_id: {cat}")
        dpg.set_value("run_scenario_video", f"video_id: {vid}")
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
        return ["docker", "compose", "exec", "-T", self.service_name] + tail_cmd

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
