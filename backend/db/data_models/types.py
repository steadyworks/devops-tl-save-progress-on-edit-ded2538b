from typing import Optional

from pydantic import BaseModel


class ExtractedExif(BaseModel):
    make: str
    model: str
    datetime_original: str
    iso: int
    exposure_time: float  # seconds
    fnumber: float  # f-stop
    focal_length: float  # mm
    gps_latitude: Optional[float] = None  # decimal degrees
    gps_longitude: Optional[float] = None  # decimal degrees


class AssetMetadata(BaseModel):
    exif_radar_formatted_address: Optional[str] = None
    exif_radar_place_label: Optional[str] = None
    exif_radar_state_code: Optional[str] = None
    exif_radar_country_code: Optional[str] = None
