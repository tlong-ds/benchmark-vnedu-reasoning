"""Base record for wire-exact intervention-run persistence."""

from pydantic import BaseModel, ConfigDict


class RecordModel(BaseModel):
    """Strict records that preserve whitespace in API evidence exactly."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=False
    )

