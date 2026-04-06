# Temporary file to show changes needed
# This file shows what needs to be changed in master_panel.py

# 1. Add enum_map_readable to __init__
# In: self.enum_values: Dict[str, List[str]] = {}
# Add after: self.enum_map_readable: Dict[Tuple[str, str], str] = {}

# 2. In _build_ui(), add viewport resize handler
# In the main render loop (run method), add: self._on_viewport_resize()

# 3. Add _parse_csv_header() method after _load_index_rows():
def _parse_csv_header(self) -> None:
    """Parse enum mappings from CSV header (weather(sunny,rainy,...)1-4)."""
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

# 4. Add _readable_enum_value() helper:
def _readable_enum_value(self, col: str, val: Any) -> str:
    """Return readable label for enum value."""
    if val is None:
        return ""
    key = (col, str(val))
    return self.enum_map_readable.get(key, str(val))

# 5. Add _on_viewport_resize() near end:
def _on_viewport_resize(self) -> None:
    """Handle viewport resize - update table sizes."""
    try:
        vp_w = dpg.get_viewport_client_width() or 1680
        vp_h = dpg.get_viewport_client_height() or 980
        table_w = max(800, vp_w - 500)
        table_h = max(500, vp_h - 30)
        dpg.configure_item("table_card", width=table_w, height=table_h)
    except Exception:
        pass

# 6. In _load_index_rows(), add at start: self._parse_csv_header()

# 7. In _reload_table_data(), call _parse_csv_header()

# 8. In _build_table_card(), change heights:
#    - table child_window: height=800 (was 750)
#    - make layout responsive

# 9. In _build_sample_conditions_card(), change height to 130 (was 165)

# 10. In _build_page_rows(), use readable values:
#     readable = self._readable_enum_value(col, val)
#     dpg.add_text("" if val is None else readable)

# 11. In _build_enum_filter_widgets(), show readable labels in combo

print("Changes needed - see code above")
