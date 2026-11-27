import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = None


class UserOut(ORM):
    id: uuid.UUID
    email: EmailStr
    full_name: str | None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


MAX_VISUAL_STYLE_LENGTH: int = 2000
STYLE_PROMPT_KEYS: tuple[str, ...] = ("visual_style", "style_prompt", "visual_direction")


def validate_visual_style_length(val: str | None) -> str | None:
    """Validate that visual direction / style prompt text does not exceed 2,000 characters."""
    if val is not None and isinstance(val, str) and len(val) > MAX_VISUAL_STYLE_LENGTH:
        raise ValueError(
            f"Style prompt / visual direction exceeds maximum allowed length of {MAX_VISUAL_STYLE_LENGTH} characters (got {len(val)})"
        )
    return val


class ChannelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    platform: Literal["youtube", "instagram"] = "youtube"
    niche: str | None = None
    handle: str | None = None
    settings: dict = {}
    default_visual_style_id: str | None = None

    @field_validator("default_visual_style_id")
    @classmethod
    def validate_default_visual_style(cls, v: str | None) -> str | None:
        if v is None:
            return None
        norm = v.strip().lower()
        if not norm:
            return None
        from .styles.registry import STYLES_REGISTRY, is_valid_style_id
        if not is_valid_style_id(norm):
            valid_ids = sorted(STYLES_REGISTRY.keys())
            raise ValueError(f"Unknown visual style '{v}'. Valid styles: {valid_ids}")
        return norm

    @field_validator("settings")
    @classmethod
    def validate_settings(cls, v: dict) -> dict:
        if isinstance(v, dict):
            for key in STYLE_PROMPT_KEYS:
                if key in v:
                    validate_visual_style_length(v[key])
        return v


class ChannelUpdate(BaseModel):
    name: str | None = None
    niche: str | None = None
    handle: str | None = None
    settings: dict | None = None
    default_visual_style_id: str | None = None

    @field_validator("default_visual_style_id")
    @classmethod
    def validate_default_visual_style(cls, v: str | None) -> str | None:
        if v is None:
            return None
        norm = v.strip().lower()
        if not norm:
            return None
        from .styles.registry import STYLES_REGISTRY, is_valid_style_id
        if not is_valid_style_id(norm):
            valid_ids = sorted(STYLES_REGISTRY.keys())
            raise ValueError(f"Unknown visual style '{v}'. Valid styles: {valid_ids}")
        return norm

    @field_validator("settings")
    @classmethod
    def validate_settings(cls, v: dict | None) -> dict | None:
        if v is not None and isinstance(v, dict):
            for key in STYLE_PROMPT_KEYS:
                if key in v:
                    validate_visual_style_length(v[key])
        return v


class ChannelOut(ORM):
    id: uuid.UUID
    name: str
    platform: str
    niche: str | None
    handle: str | None
    settings: dict
    default_visual_style_id: str | None = None
    created_at: datetime


class VideoCreate(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    description: str | None = None
    visual_style_id: str = Field(..., min_length=1, description="Explicit catalog visual style ID")

    @field_validator("visual_style_id")
    @classmethod
    def validate_visual_style(cls, v: str) -> str:
        if v is None:
            raise ValueError("visual_style_id is required")
        norm = v.strip().lower()
        if not norm:
            raise ValueError("visual_style_id must not be empty")
        from .styles.registry import STYLES_REGISTRY, is_valid_style_id
        if not is_valid_style_id(norm):
            valid_ids = sorted(STYLES_REGISTRY.keys())
            raise ValueError(f"Unknown visual style '{v}'. Valid styles: {valid_ids}")
        return norm


class VideoStyleUpdate(BaseModel):
    visual_style_id: str

    @field_validator("visual_style_id")
    @classmethod
    def validate_visual_style(cls, v: str) -> str:
        norm = v.strip().lower()
        from .styles.registry import STYLES_REGISTRY, is_valid_style_id
        if not is_valid_style_id(norm):
            valid_ids = sorted(STYLES_REGISTRY.keys())
            raise ValueError(f"Unknown visual style '{v}'. Valid styles: {valid_ids}")
        return norm


class VisualStyleSummaryOut(BaseModel):
    id: str
    name: str
    version: int
    category: str
    description: str
    preview_image_path: str
    is_default: bool = False


class VisualStylesListOut(BaseModel):
    default_style_id: str
    styles: list[VisualStyleSummaryOut]


class SubtitleStyleOut(BaseModel):
    id: str
    name: str
    description: str
    is_default: bool = False


class StepOut(ORM):
    name: str
    position: int
    status: str
    output: dict | None
    user_prompt: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    revisions_count: int = 0

    @field_validator("output", mode="after")
    @classmethod
    def ensure_subtitles_style(cls, v: dict | None) -> dict | None:
        if isinstance(v, dict) and "subtitles" in v and isinstance(v["subtitles"], dict):
            if not v["subtitles"].get("style"):
                new_v = dict(v)
                new_sub = dict(new_v["subtitles"])
                new_sub["style"] = "clean_cinematic"
                new_v["subtitles"] = new_sub
                return new_v
        return v


class StepRevisionOut(ORM):
    id: uuid.UUID
    step_id: uuid.UUID
    version: int
    output: dict | None
    user_prompt: str | None
    change_summary: str | None
    created_at: datetime


class StepApproveBody(BaseModel):
    auto_advance: bool = False


class StepContentUpdate(BaseModel):
    output: dict
    change_summary: str | None = None

    @field_validator("output")
    @classmethod
    def validate_output(cls, v: dict) -> dict:
        if isinstance(v, dict):
            for key in STYLE_PROMPT_KEYS:
                if key in v:
                    validate_visual_style_length(v[key])
        return v


class JobOut(ORM):
    id: uuid.UUID
    video_id: uuid.UUID
    step_name: str
    status: str
    run_all: bool
    attempts: int
    max_attempts: int
    scheduled_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    worker_id: str | None
    error: str | None



class VideoSummary(ORM):
    id: uuid.UUID
    channel_id: uuid.UUID
    topic: str
    description: str | None
    status: str
    visual_style_id: str | None = None
    visual_style_version: int | None = None
    created_at: datetime


class VideoDetail(VideoSummary):
    visual_style_snapshot: dict | None = None
    steps: list[StepOut] = []


class RerunBody(BaseModel):
    prompt: str | None = None  # "make the hook more dramatic", etc.
    subtitle_style: str | None = None

    @field_validator("subtitle_style")
    @classmethod
    def validate_subtitle_style(cls, v: str | None) -> str | None:
        if v is None:
            return None
        norm = v.strip().lower()
        from .subtitles.styles import SUBTITLE_STYLES
        if norm not in SUBTITLE_STYLES:
            valid_ids = sorted(SUBTITLE_STYLES.keys())
            raise ValueError(f"Unknown subtitle style '{v}'. Valid styles: {valid_ids}")
        return norm


# ============================================================================
# Creative Direction & Video Director Schemas (Phase 1)
# ============================================================================

class EmotionalBeat(BaseModel):
    phase: Literal["hook", "setup", "complication", "reveal", "resolution"]
    target_emotion: str = Field(description="Target emotional tone, e.g. curiosity, drowsiness, abrupt vertigo, relief")
    visual_intent: str = Field(description="Directorial note on how visuals evoke this emotion")


class ProtagonistConfig(BaseModel):
    human_presence: Literal["none", "contextual", "single", "ensemble"] = Field(
        description="Human presence level: 'none' (purely inanimate/scientific), 'contextual' (witnesses/workers/communities without recurring hero), 'single' (featured recurring protagonist), 'ensemble' (featured recurring ensemble).",
    )
    archetype_summary: str | None = Field(
        default=None,
        description="Summary of character or contextual witness role; None if human_presence is none",
    )


class CreativeDirectionOutput(BaseModel):
    format_archetype: str = Field(
        description="Story format archetype, e.g. explainer_myth_bust, narrative_story, mechanical_breakdown",
    )
    storytelling_angle: str
    central_thesis: str
    tone: str
    protagonist_config: ProtagonistConfig
    emotional_progression: list[EmotionalBeat] = Field(default_factory=list)
    creative_constraints: list[str] = Field(default_factory=list)


class CharacterEntity(BaseModel):
    id: str
    name: str
    role: str
    visual_anchor: str = Field(description="Physical appearance: age, face, hair, body, materials")
    apparel_anchor: str = Field(description="Exact clothing items, colors, and textures")
    forbidden_mutations: list[str] = Field(default_factory=list, description="Specific elements that must never change")


class EnvironmentEntity(BaseModel):
    id: str
    name: str
    spatial_layout: str = Field(description="Stable macro spatial layout, boundary, and landscape/architectural geometry")
    lighting_setup: str = Field(description="Key light source, ambient color temperature, atmospheric mood")
    persistent_props: list[str] = Field(default_factory=list, description="Permanent world props associated with this environment")
    time_of_day: str = "unspecified"
    environment_type: Literal["exterior", "interior", "atmospheric", "macro_surface", "conceptual_void"] = "exterior"


class ObjectEntity(BaseModel):
    id: str
    name: str
    visual_description: str
    material_and_finish: str


class ConceptEntity(BaseModel):
    id: str
    name: str
    concept_explained: str
    visual_analogy: str
    graphical_elements: list[str] = Field(default_factory=list)


class EntityCatalog(BaseModel):
    characters: list[CharacterEntity] = Field(default_factory=list)
    environments: list[EnvironmentEntity] = Field(default_factory=list)
    objects: list[ObjectEntity] = Field(default_factory=list)
    concepts: list[ConceptEntity] = Field(default_factory=list)


class ArtDirection(BaseModel):
    style_family: str
    rendering_engine_flavor: str = "default"
    primary_palette_hex: list[str] = Field(default_factory=list)
    negative_prompt_tokens: str = ""
    visual_genre: str = "photorealistic_cinematic"
    lighting_philosophy: str = ""
    rendering_texture: str = ""
    negative_anchors: list[str] = Field(default_factory=list)


class CinematographyProfile(BaseModel):
    aspect_ratio: str = "9:16 vertical"
    subject_safe_zone: str = "upper 55% of viewport"
    caption_safe_zone: str = "lower 35% of viewport"


class FoleySound(BaseModel):
    id: str
    cue_name: str
    asset_file: str


class AudioDirection(BaseModel):
    voice_identifier: str = "am_adam"
    default_speech_speed: float = 1.0
    ambient_bed_track: str | None = None
    music_bed_track: str | None = None
    music_duck_db: float = -15.0
    foley_palette: list[FoleySound] = Field(default_factory=list)


class SubtitleProfile(BaseModel):
    font_family: str = "Montserrat ExtraBold"
    font_size_pt: int = 52
    primary_color_hex: str = "#FFFFFF"
    highlight_color_hex: str = "#F7931E"
    margin_vertical_px: int = 280


class VideoBibleOutput(BaseModel):
    bible_id: str
    topic_summary: str
    entities: EntityCatalog = Field(default_factory=EntityCatalog)
    art_direction: ArtDirection = Field(default_factory=lambda: ArtDirection(style_family="cinematic"))
    cinematography: CinematographyProfile = Field(default_factory=CinematographyProfile)
    audio_direction: AudioDirection = Field(default_factory=AudioDirection)
    subtitle_direction: SubtitleProfile = Field(default_factory=SubtitleProfile)


class EntityBindings(BaseModel):
    character_id: str | None = None
    secondary_character_ids: list[str] = Field(default_factory=list)
    environment_id: str | None = None
    object_ids: list[str] = Field(default_factory=list)
    machine_ids: list[str] = Field(default_factory=list)
    concept_id: str | None = None


class CameraDirection(BaseModel):
    shot_type: Literal[
        "extreme_close_up",
        "close_up",
        "medium_close_up",
        "medium",
        "wide",
        "extreme_wide",
    ] = "medium"
    camera_angle: Literal[
        "eye_level",
        "low_angle",
        "high_angle",
        "top_down",
        "side_profile",
        "dutch_angle",
    ] = "eye_level"
    focal_point_description: str = ""
    movement_type: Literal[
        "static",
        "push_in",
        "pull_out",
        "pan_left",
        "pan_right",
        "drift",
    ] = "static"


class CharacterState(BaseModel):
    pose_description: str
    expression_description: str
    action_in_progress: str


class FoleyCue(BaseModel):
    foley_id: str
    offset_seconds: float = 0.0
    relative_volume: float = 1.0


VisualIntentType = Literal[
    "establishing",
    "action_event",
    "environmental_context",
    "mechanism_explanation",
    "human_experience",
    "detail_focus",
    "abstract_visualization",
    "narrative_resolution",
]

VisualTechniqueType = Literal[
    "cinematic_illustration",
    "physical_close_up",
    "in_scene_cue",
    "cutaway",
    "diagram",
]


class StoryboardBeat(BaseModel):
    beat_id: str
    scene_index: int
    order: int
    narration_slice: str
    duration_est_seconds: float
    visual_intent: VisualIntentType = Field(description="Required structured visual intent")
    visual_technique: VisualTechniqueType = Field(
        default="cinematic_illustration",
        description="Canonical visual presentation technique",
    )
    entity_bindings: EntityBindings
    transition_from_previous: Literal[
        "establishing",
        "same_scene_reframe",
        "action_reaction",
        "spatial_cutaway",
        "location_change",
        "explanatory_cutaway",
        "surreal_metaphor",
        "resolution_return",
    ] = "establishing"
    continuity_intent: str = ""
    camera: CameraDirection = Field(default_factory=CameraDirection)
    character_state: CharacterState | None = None
    environment_state_override: str | None = None
    foley_cues: list[FoleyCue] = Field(default_factory=list)
    final_prompt: str | None = None


class StoryboardOutput(BaseModel):
    beats: list[StoryboardBeat] = Field(default_factory=list)



