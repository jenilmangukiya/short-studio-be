"""Subtitle visual style registry and definitions for Shorts Studio.

Provides a centralized, typed registry of visual subtitle styles used for ASS rendering
and captioned video composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubtitleStyle:
    """Configuration defining the visual appearance and ASS styling of subtitles.

    Attributes:
        id: Unique identifier for the style (e.g. 'clean_cinematic').
        name: Human-readable display name.
        description: Short description of the style aesthetic.
        font_name: Font family name used in ASS (e.g. 'Arial').
        font_size: Font size in points for 1080x1920 portrait resolution.
        bold: Whether font is rendered in bold weight.
        primary_color_ass: Inactive text color in ASS BGR format (&HAABBGGRR&).
        secondary_color_ass: Secondary color in ASS BGR format (&HAABBGGRR&).
        outline_color_ass: Border/stroke color in ASS BGR format (&HAABBGGRR&).
        back_color_ass: Shadow or box background color in ASS BGR format (&HAABBGGRR&).
        outline_width: Thickness of outline border in pixels.
        shadow_depth: Depth of drop shadow in pixels.
        border_style: ASS BorderStyle (1 = Outline with shadow, 3 = Opaque/translucent box).
        alignment: ASS Alignment (2 = Bottom-Center).
        margin_l: Left margin in pixels.
        margin_r: Right margin in pixels.
        margin_v: Vertical margin from bottom edge in pixels.
        active_color_ass: Highlight color for currently spoken word in ASS BGR format (&HAABBGGRR&).
        active_scale_percent: Active word scale percentage (e.g. 106 for 106%, None or 100 for no scaling).
        uppercase: Whether caption text should be transformed to UPPERCASE.
    """

    id: str
    name: str
    description: str
    font_name: str = "Arial"
    font_size: int = 64
    bold: bool = True
    primary_color_ass: str = "&H00FFFFFF&"
    secondary_color_ass: str = "&H000000FF&"
    outline_color_ass: str = "&H00000000&"
    back_color_ass: str = "&H80000000&"
    outline_width: float = 4.0
    shadow_depth: float = 2.0
    border_style: int = 1
    alignment: int = 2
    margin_l: int = 40
    margin_r: int = 40
    margin_v: int = 360
    active_color_ass: str = "&H0028F5FC&"
    active_scale_percent: int | None = None
    uppercase: bool = False

    @property
    def primary_color_header(self) -> str:
        """Primary color formatted for ASS [V4+ Styles] header (no trailing '&')."""
        return self.primary_color_ass.rstrip("&")

    @property
    def secondary_color_header(self) -> str:
        """Secondary color formatted for ASS [V4+ Styles] header (no trailing '&')."""
        return self.secondary_color_ass.rstrip("&")

    @property
    def outline_color_header(self) -> str:
        """Outline color formatted for ASS [V4+ Styles] header (no trailing '&')."""
        return self.outline_color_ass.rstrip("&")

    @property
    def back_color_header(self) -> str:
        """Shadow/back color formatted for ASS [V4+ Styles] header (no trailing '&')."""
        return self.back_color_ass.rstrip("&")

    @property
    def active_color_tag(self) -> str:
        """ASS inline override tag for active color (e.g. '\\c&H00FCF528&')."""
        c = self.active_color_ass.rstrip("&")
        return f"\\c{c}&"

    @property
    def primary_color_tag(self) -> str:
        """ASS inline override tag for primary/reset color (e.g. '\\c&H00FFFFFF&')."""
        c = self.primary_color_ass.rstrip("&")
        return f"\\c{c}&"

    def build_ass_style_header(self, style_name: str = "Default") -> str:
        """Format the ASS [V4+ Styles] line for this style."""
        bold_val = -1 if self.bold else 0
        outline_str = f"{self.outline_width:g}"
        shadow_str = f"{self.shadow_depth:g}"
        return (
            f"Style: {style_name},{self.font_name},{self.font_size},"
            f"{self.primary_color_header},{self.secondary_color_header},"
            f"{self.outline_color_header},{self.back_color_header},"
            f"{bold_val},0,0,0,100,100,0,0,{self.border_style},"
            f"{outline_str},{shadow_str},{self.alignment},"
            f"{self.margin_l},{self.margin_r},{self.margin_v},1"
        )


# ---------------------------------------------------------------------------
# Predefined Subtitle Styles
# ---------------------------------------------------------------------------

# A. Clean Cinematic (Default)
# Translucent modern box matching the frontend preview (Image 2):
# 60px bold typography inside a 65% opacity dark translucent box (BorderStyle: 3),
# with amber-300 active word highlight (#FCD34D -> &H004DD3FC& in ASS BGR) and 108% scale pop.
CLEAN_CINEMATIC = SubtitleStyle(
    id="clean_cinematic",
    name="Clean Cinematic",
    description="Clean, elegant subtitle style with a translucent dark box and amber active-word highlight.",
    font_name="Arial",
    font_size=60,
    bold=True,
    primary_color_ass="&H00FFFFFF&",
    secondary_color_ass="&H000000FF&",
    outline_color_ass="&H5A000000&",
    back_color_ass="&H5A000000&",
    outline_width=14.0,
    shadow_depth=0.0,
    border_style=3,
    alignment=2,
    margin_l=40,
    margin_r=40,
    margin_v=360,
    active_color_ass="&H004DD3FC&",
    active_scale_percent=108,
    uppercase=False,
)

# B. Remotion Neon
# Modeled after the Remotion POC: 58px bold typography, electric cyan active word
# (#28F5FC -> &H00FCF528& in ASS BGR order), 106% scale pop, inside translucent dark box (BorderStyle: 3).
REMOTION_NEON = SubtitleStyle(
    id="remotion_neon",
    name="Remotion Neon",
    description="High-energy viral Shorts style inspired by the Remotion POC with electric cyan active-word pop and translucent capsule box.",
    font_name="Arial",
    font_size=58,
    bold=True,
    primary_color_ass="&H00FFFFFF&",
    secondary_color_ass="&H000000FF&",
    outline_color_ass="&H5A000000&",
    back_color_ass="&H5A000000&",
    outline_width=14.0,
    shadow_depth=0.0,
    border_style=3,
    alignment=2,
    margin_l=40,
    margin_r=40,
    margin_v=340,
    active_color_ass="&H00FCF528&",  # #28F5FC in ASS BGR (&HAABBGGRR&)
    active_scale_percent=106,
    uppercase=False,
)

# C. Dynamic Highlight
# High-impact creator typography: 64px bold, uppercase caption text, vibrant yellow
# active word (#FFE600 -> &H0000E6FF& in ASS BGR order), 110% scale punch, inside translucent dark box (BorderStyle: 3).
DYNAMIC_HIGHLIGHT = SubtitleStyle(
    id="dynamic_highlight",
    name="Dynamic Highlight",
    description="Punchy, creator-focused all-caps typography with vibrant yellow active-word punch and translucent capsule box.",
    font_name="Arial",
    font_size=64,
    bold=True,
    primary_color_ass="&H00FFFFFF&",
    secondary_color_ass="&H000000FF&",
    outline_color_ass="&H5A000000&",
    back_color_ass="&H5A000000&",
    outline_width=14.0,
    shadow_depth=0.0,
    border_style=3,
    alignment=2,
    margin_l=40,
    margin_r=40,
    margin_v=360,
    active_color_ass="&H0000E6FF&",  # #FFE600 in ASS BGR (&HAABBGGRR&)
    active_scale_percent=110,
    uppercase=True,
)


DEFAULT_SUBTITLE_STYLE_ID = "clean_cinematic"

SUBTITLE_STYLES: dict[str, SubtitleStyle] = {
    CLEAN_CINEMATIC.id: CLEAN_CINEMATIC,
    REMOTION_NEON.id: REMOTION_NEON,
    DYNAMIC_HIGHLIGHT.id: DYNAMIC_HIGHLIGHT,
}


def get_subtitle_style(style_id: str | None = None) -> SubtitleStyle:
    """Retrieve a subtitle style by ID.

    If style_id is None, empty, or unrecognised, safely falls back to the default
    'clean_cinematic' style to guarantee backward compatibility.
    """
    if not style_id:
        return SUBTITLE_STYLES[DEFAULT_SUBTITLE_STYLE_ID]
    norm = str(style_id).strip().lower()
    return SUBTITLE_STYLES.get(norm, SUBTITLE_STYLES[DEFAULT_SUBTITLE_STYLE_ID])


def list_subtitle_styles() -> list[SubtitleStyle]:
    """Return all registered subtitle styles in canonical display order."""
    return list(SUBTITLE_STYLES.values())
