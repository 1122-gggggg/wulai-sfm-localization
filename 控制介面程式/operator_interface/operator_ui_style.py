"""Shared desktop palette; styles belong to each Tk interpreter, never a global root."""
from __future__ import annotations

from tkinter import ttk

# Warm 米黃 surfaces, lightest (inputs/cards) to darkest (pressed controls).
BG = "#ede3c8"          # window gutters between cards
PANEL = "#f8f1df"       # cards and default content surface
CARD = "#fffaee"        # raised chips / read-only cards placed on a panel
FIELD = "#fffdf7"       # text inputs, lists, read-only output
SURFACE = "#efe3c2"     # secondary buttons, table headers
HOVER = "#e6d4a4"
PRESSED = "#d8c18a"
BORDER = "#e1d2aa"      # hairlines
BORDER_STRONG = "#c7b07a"  # control outlines
TEXT = "#2e2516"
MUTED = "#6b5b3e"
DISABLED = "#a09175"
HEADING = "#7a4a0a"     # section titles (7:1 on PANEL)
ACCENT = "#9a5c08"
ACCENT_HOVER = "#b06a0a"
SELECT_BG = "#ebcf86"
SELECT_FG = "#2b2112"

# Espresso title band: the one dark surface outside the viewports.
HEADER_BG = "#3a2e1e"
HEADER_FG = "#f6ecd2"
HEADER_MUTED = "#cdbb90"
HEADER_ACCENT = "#e3a640"
HEADER_SURFACE = "#54442c"
HEADER_HOVER = "#665337"
HEADER_BORDER = "#7a6542"

# Semantic fills carry white text (>= 4.5:1); the text colours are for beige.
STATE_NEUTRAL = "#66573c"
STATE_WAITING = "#7a6a49"
STATE_GOOD = "#2b6a45"
STATE_WARNING = "#8a5a12"
STATE_BLOCKED = "#a1322a"
TEXT_GOOD = "#1f6b3a"
TEXT_WARNING = "#8a5a00"
TEXT_BLOCKED = "#a1322a"
ON_STATE = "#ffffff"

# Viewports (video, 3D map, route point cloud) stay dark by design.
VIEWPORT_MAP_BG = "#15181c"
VIEWPORT_DARK_BG = "#08090b"
VIEWPORT_EDITOR_BG = "#121519"
# Transparent (outline + text only) controls drawn on the dark map viewport.
MAP_GHOST_FG = "#f1e6c8"
MAP_GHOST_FG_HOVER = "#ffffff"
MAP_GHOST_BORDER = "#b8a572"
# Opaque overlays (zoom buttons, readouts) floating on a dark viewport.
MAP_OVERLAY_BG = "#22262c"
MAP_OVERLAY_HOVER = "#30353d"
MAP_OVERLAY_PRESSED = "#3b414a"

FONT_FAMILY = "Sans"
FONT_BODY = (FONT_FAMILY, 11)
FONT_SMALL = (FONT_FAMILY, 10)
FONT_BOLD = (FONT_FAMILY, 11, "bold")
FONT_SMALL_BOLD = (FONT_FAMILY, 10, "bold")
FONT_TITLE = (FONT_FAMILY, 12, "bold")
FONT_DISPLAY = (FONT_FAMILY, 14, "bold")

# Padding rhythm shared by cards and dialogs: 4 / 8 / 12 pixels.
PAD_S, PAD_M, PAD_L = 4, 8, 12

# Options shared by every read-only Text / Listbox so they match the ttk Entry look
# (1px hairline that darkens on focus). Padding is chosen by the caller.
TEXT_BOX = {
    "background": FIELD, "foreground": TEXT, "relief": "flat", "borderwidth": 0,
    "highlightthickness": 1, "highlightbackground": BORDER_STRONG,
    "highlightcolor": ACCENT,
}

#: (style prefix, normal, active, pressed). Every fill carries white text.
_SOLID_BUTTONS = (
    ("Flight", "#6e5b3a", "#7d6844", "#5a4a2f"),
    ("Hover", "#805016", "#8a5a17", "#6a420f"),
    ("Land", "#a4481c", "#a84b1b", "#8a3a14"),
    ("EmergencyStop", "#b42318", "#bb2419", "#9a1c12"),
    ("Takeoff", "#237a45", "#1f6f3d", "#1a5e33"),
    ("Auto", "#245b9b", "#2864a8", "#1d4c82"),
    ("Primary", ACCENT, "#874f06", "#723f03"),
)


def _flat_button(style, name: str, *, normal: str, active: str, pressed: str,
                 foreground: str, border: str, disabled_bg: str, disabled_fg: str,
                 disabled_border: str, focus: str, font=FONT_BODY,
                 padding=(10, 4)) -> None:
    """One flat clam button: a 1px outline, no bevel, fill changes on hover/press.

    clam draws the outline only for a raised/sunken relief; matching the bevel
    colours to the fill is what keeps it flat.
    """
    style.configure(name, background=normal, foreground=foreground, font=font,
                    padding=padding, relief="raised", borderwidth=1,
                    bordercolor=border, lightcolor=normal, darkcolor=normal,
                    focuscolor=focus, focusthickness=1, anchor="center")
    style.map(name,
              background=[("disabled", disabled_bg), ("pressed", pressed), ("active", active)],
              foreground=[("disabled", disabled_fg), ("!disabled", foreground)],
              bordercolor=[("disabled", disabled_border), ("focus", focus)],
              lightcolor=[("disabled", disabled_bg), ("pressed", pressed), ("active", active)],
              darkcolor=[("disabled", disabled_bg), ("pressed", pressed), ("active", active)],
              relief=[("pressed", "raised")])


def _slim_scrollbars(style) -> None:
    """Arrowless, thin thumbs: the content, not the chrome, should draw the eye."""
    for orient, sticky in (("Vertical", "ns"), ("Horizontal", "we")):
        style.layout(f"{orient}.TScrollbar", [(
            f"{orient}.Scrollbar.trough", {"sticky": sticky, "children": [
                (f"{orient}.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"}),
            ]},
        )])
        style.configure(f"{orient}.TScrollbar", background=BORDER_STRONG,
                        troughcolor=PANEL, bordercolor=PANEL, lightcolor=BORDER_STRONG,
                        darkcolor=BORDER_STRONG, arrowsize=10, gripcount=0, relief="flat")
        style.map(f"{orient}.TScrollbar",
                  background=[("disabled", BORDER), ("pressed", ACCENT),
                              ("active", MUTED)],
                  lightcolor=[("disabled", BORDER), ("pressed", ACCENT), ("active", MUTED)],
                  darkcolor=[("disabled", BORDER), ("pressed", ACCENT), ("active", MUTED)])


def apply_operator_styles(root) -> None:
    """Apply the same readable, non-animated theme to the operator and editors."""
    style = ttk.Style(root)
    if style.theme_use() != "clam":
        style.theme_use("clam")
    root.configure(background=BG)
    root.option_add("*Font", FONT_BODY)
    style.configure(".", font=FONT_BODY, background=PANEL, foreground=TEXT,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                    troughcolor=FIELD, focuscolor=ACCENT, selectbackground=SELECT_BG,
                    selectforeground=SELECT_FG)

    # ---- surfaces -----------------------------------------------------------
    for name in ("TFrame", "Panel.TFrame"):
        style.configure(name, background=PANEL)
    style.configure("Gutter.TFrame", background=BG)
    style.configure("Card.TFrame", background=PANEL, relief="solid", borderwidth=1,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("Header.TFrame", background=HEADER_BG)
    style.configure("MapOverlay.TFrame", background=MAP_OVERLAY_BG)

    # ---- text ---------------------------------------------------------------
    style.configure("TLabel", background=PANEL, foreground=TEXT)
    style.configure("Secondary.TLabel", foreground=MUTED, font=FONT_SMALL)
    style.configure("Heading.TLabel", foreground=HEADING, font=FONT_BOLD)
    style.configure("Title.TLabel", foreground=TEXT, font=FONT_TITLE)
    style.configure("Display.TLabel", foreground=TEXT, font=FONT_DISPLAY)
    style.configure("Status.TLabel", background=FIELD, foreground=TEXT, padding=(8, 4))
    style.configure("Card.TLabel", background=CARD, foreground=TEXT, padding=(8, 6))
    style.configure("Gutter.TLabel", background=BG, foreground=MUTED, font=FONT_SMALL)
    style.configure("Error.TLabel", foreground=TEXT_BLOCKED, font=FONT_SMALL)
    style.configure("Header.TLabel", background=HEADER_BG, foreground=HEADER_FG,
                    font=FONT_TITLE)
    style.configure("HeaderTitle.TLabel", background=HEADER_BG, foreground=HEADER_FG,
                    font=FONT_DISPLAY)
    style.configure("HeaderBrand.TLabel", background=HEADER_BG, foreground=HEADER_ACCENT,
                    font=(FONT_FAMILY, 15, "bold"))
    style.configure("HeaderMuted.TLabel", background=HEADER_BG, foreground=HEADER_MUTED,
                    font=FONT_SMALL)
    style.configure("MapOverlay.TLabel", background=MAP_OVERLAY_BG, foreground=MAP_GHOST_FG,
                    font=FONT_SMALL, padding=(6, 3))

    # ---- buttons ------------------------------------------------------------
    _flat_button(style, "TButton", normal=SURFACE, active=HOVER, pressed=PRESSED,
                 foreground=TEXT, border=BORDER_STRONG, disabled_bg=PANEL,
                 disabled_fg=DISABLED, disabled_border=BORDER, focus=ACCENT)
    for name, normal, active, pressed in _SOLID_BUTTONS:
        _flat_button(style, f"{name}.TButton", normal=normal, active=active,
                     pressed=pressed, foreground=ON_STATE, border=pressed,
                     disabled_bg=PANEL, disabled_fg=DISABLED, disabled_border=BORDER,
                     focus=TEXT, font=FONT_BOLD)
    # Quiet tertiary action: reads as a link until hovered.
    _flat_button(style, "Ghost.TButton", normal=PANEL, active=SURFACE, pressed=HOVER,
                 foreground=HEADING, border=PANEL, disabled_bg=PANEL,
                 disabled_fg=DISABLED, disabled_border=PANEL, focus=ACCENT,
                 padding=(8, 4))
    # Controls that live on the espresso title band.
    _flat_button(style, "Header.TButton", normal=HEADER_SURFACE, active=HEADER_HOVER,
                 pressed=HEADER_BORDER, foreground=HEADER_FG, border=HEADER_BORDER,
                 disabled_bg=HEADER_BG, disabled_fg=HEADER_MUTED,
                 disabled_border=HEADER_SURFACE, focus=HEADER_ACCENT)
    _flat_button(style, "HeaderPrimary.TButton", normal=ACCENT, active="#874f06",
                 pressed="#723f03", foreground=ON_STATE, border=HEADER_ACCENT,
                 disabled_bg=HEADER_BG, disabled_fg=HEADER_MUTED,
                 disabled_border=HEADER_SURFACE, focus=HEADER_FG, font=FONT_BOLD)
    # Dark viewport overlays (map / editor zoom): opaque, but tuned to the canvas.
    _flat_button(style, "MapZoom.TButton", normal=MAP_OVERLAY_BG, active=MAP_OVERLAY_HOVER,
                 pressed=MAP_OVERLAY_PRESSED, foreground=MAP_GHOST_FG,
                 border=MAP_GHOST_BORDER, disabled_bg=MAP_OVERLAY_BG,
                 disabled_fg=DISABLED, disabled_border=MAP_OVERLAY_HOVER,
                 focus=MAP_GHOST_FG_HOVER, font=(FONT_FAMILY, 15, "bold"),
                 padding=(8, 10))

    # ---- grouped content ------------------------------------------------------
    style.configure("TLabelframe", background=PANEL, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER, relief="solid", borderwidth=1,
                    labelmargins=(10, 2, 10, 2))
    style.configure("TLabelframe.Label", background=PANEL, foreground=HEADING,
                    font=FONT_BOLD)
    for name in ("TCheckbutton", "TRadiobutton"):
        style.configure(name, background=PANEL, foreground=TEXT, padding=(2, 2),
                        indicatorbackground=FIELD, indicatorforeground=TEXT,
                        upperbordercolor=BORDER_STRONG, lowerbordercolor=BORDER_STRONG,
                        indicatormargin=(1, 1, 6, 1))
        style.map(name,
                  background=[("active", PANEL)],
                  foreground=[("disabled", DISABLED), ("active", HEADING)],
                  indicatorbackground=[("disabled", PANEL), ("pressed", HOVER),
                                       ("selected", ACCENT), ("alternate", FIELD),
                                       ("!selected", FIELD)],
                  indicatorforeground=[("disabled", DISABLED), ("selected", ON_STATE),
                                       ("alternate", ACCENT)],
                  upperbordercolor=[("disabled", BORDER), ("selected", ACCENT),
                                    ("alternate", ACCENT), ("active", ACCENT)],
                  lowerbordercolor=[("disabled", BORDER), ("selected", ACCENT),
                                    ("alternate", ACCENT), ("active", ACCENT)])
    for name in ("TEntry", "TSpinbox", "TCombobox"):
        style.configure(name, fieldbackground=FIELD, background=SURFACE,
                        foreground=TEXT, insertcolor=TEXT, arrowcolor=MUTED,
                        bordercolor=BORDER_STRONG, lightcolor=FIELD, darkcolor=FIELD,
                        padding=(7, 4), arrowsize=13)
        style.map(name, fieldbackground=[("disabled", PANEL), ("readonly", FIELD)],
                  foreground=[("disabled", DISABLED), ("readonly", TEXT)],
                  bordercolor=[("disabled", BORDER), ("focus", ACCENT), ("hover", MUTED)],
                  lightcolor=[("focus", FIELD)], darkcolor=[("focus", FIELD)],
                  arrowcolor=[("disabled", DISABLED), ("active", ACCENT)],
                  background=[("disabled", PANEL), ("pressed", PRESSED),
                              ("active", HOVER), ("readonly", FIELD)],
                  selectbackground=[("!disabled", SELECT_BG)],
                  selectforeground=[("!disabled", SELECT_FG)])
    for orient in ("Horizontal", "Vertical"):
        style.configure(f"{orient}.TScale", background=ACCENT, troughcolor=BORDER,
                        bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                        sliderlength=16, sliderthickness=16, troughrelief="flat")
        style.map(f"{orient}.TScale",
                  background=[("disabled", BORDER_STRONG), ("active", ACCENT_HOVER)],
                  bordercolor=[("disabled", BORDER_STRONG)],
                  lightcolor=[("disabled", BORDER_STRONG), ("active", ACCENT_HOVER)],
                  darkcolor=[("disabled", BORDER_STRONG), ("active", ACCENT_HOVER)])
        style.configure(f"{orient}.TProgressbar", background=ACCENT, troughcolor=BORDER,
                        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
    _slim_scrollbars(style)
    style.configure("TSeparator", background=BORDER)
    style.configure("Header.TSeparator", background=HEADER_BORDER)
    style.configure("Sash", sashthickness=6, gripcount=0, background=BG,
                    lightcolor=BG, darkcolor=BG)
    style.configure("Treeview", background=FIELD, fieldbackground=FIELD,
                    foreground=TEXT, bordercolor=BORDER_STRONG, lightcolor=FIELD,
                    darkcolor=FIELD, rowheight=28, padding=1)
    style.map("Treeview", background=[("selected", SELECT_BG)],
              foreground=[("selected", SELECT_FG), ("disabled", DISABLED)],
              bordercolor=[("focus", ACCENT)])
    style.configure("Route.Treeview", rowheight=24)
    style.configure("Treeview.Heading", background=SURFACE, foreground=MUTED,
                    font=FONT_SMALL_BOLD, relief="flat", padding=(8, 5),
                    bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE)
    style.map("Treeview.Heading", background=[("active", HOVER)],
              lightcolor=[("active", HOVER)], darkcolor=[("active", HOVER)])
    style.configure("TNotebook", background=PANEL, borderwidth=1, bordercolor=BORDER,
                    lightcolor=PANEL, darkcolor=PANEL, tabmargins=(4, 6, 4, 0))
    style.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                    padding=(14, 5), font=FONT_BOLD, bordercolor=BORDER,
                    lightcolor=BG, darkcolor=BG)
    style.map("TNotebook.Tab",
              background=[("selected", PANEL), ("active", SURFACE)],
              foreground=[("selected", HEADING), ("active", TEXT)],
              bordercolor=[("selected", BORDER_STRONG)],
              lightcolor=[("selected", ACCENT), ("active", SURFACE)],
              darkcolor=[("selected", PANEL), ("active", SURFACE)],
              expand=[("selected", (0, 3, 0, 0))])
    # Keep preflight navigation without a redundant single-tab strip.
    style.layout("Workspace.TNotebook.Tab", [])
    style.configure("Workspace.TNotebook", background=BG, bordercolor=BORDER,
                    lightcolor=PANEL, darkcolor=PANEL, tabmargins=0)

    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", SELECT_BG)
    root.option_add("*TCombobox*Listbox.selectForeground", SELECT_FG)
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    root.option_add("*TCombobox*Listbox.font", FONT_BODY)
    # Plain Tk widgets (lists, read-only output, Tk's own file dialogs) follow the
    # same surfaces and selection colours.
    for widget in ("Listbox", "Text", "Entry", "Spinbox"):
        root.option_add(f"*{widget}.selectBackground", SELECT_BG)
        root.option_add(f"*{widget}.selectForeground", SELECT_FG)
        root.option_add(f"*{widget}.insertBackground", TEXT)
    root.option_add("*Toplevel.background", BG)
    root.option_add("*Menu.background", PANEL)
    root.option_add("*Menu.foreground", TEXT)
    root.option_add("*Menu.activeBackground", SELECT_BG)
    root.option_add("*Menu.activeForeground", SELECT_FG)
