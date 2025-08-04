from pydantic import BaseModel,Field
from typing import List, Optional, Dict 

class TextData(BaseModel):
    reference: str = Field(..., example="frog jumps", description="The reference text to compare against.")
    hypothesis: Optional[str] = Field(None, example="dog jumps", description="The hypothesis text to be compared.")
    language: str = Field(..., example="en", description="The language of the text.")
    base64_string: Optional[str] = Field(None, example="", description="Base64-encoded WAV audio.")

class audioData(BaseModel):
    base64_string: str = Field(..., example="UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAABCxAgAEABAAZGF0YUAA", description="Base64 encoded audio string.")
    enablePauseCount: bool = Field(..., example=True, description="Flag to enable pause count detection.")
    enableDenoiser: bool = Field(..., example=True, description="Flag to enable audio denoising.")
    enable_prosody_fluency: bool = Field(False,example=True,description="If True, run pitch, intensity, expression, and smoothness classification.")
    contentType: str = Field(..., example="Word", description="The type of content in the audio.")

class PhonemesRequest(BaseModel):
    text: str = Field(..., example="dog jumps", description="The text to convert into phonemes.")

class PhonemesResponse(BaseModel):
    phonemes: List[str] = Field(..., example=["d", "ɔ", "g", "ʤ", "ə", "m", "p", "s"], description="List of phonemes extracted from the text.")

class ErrorArraysResponse(BaseModel):
    wer: float = Field(..., example=0.5, description="Word Error Rate.")
    cer: float = Field(..., example=0.2, description="Character Error Rate.")
    insertion: List[str] = Field(..., example=[], description="List of insertions.")
    insertion_count: int = Field(..., example=0, description="Count of insertions.")
    deletion: List[str] = Field(..., example=["r"], description="List of deletions.")
    deletion_count: int = Field(..., example=1, description="Count of deletions.")
    substitution: List[Dict[str, str]] = Field(..., example=[{"removed": "d", "replaced": "f"}], description="List of substitutions.")
    substitution_count: int = Field(..., example=1, description="Count of substitutions.")
    pause_count: Optional[int] = Field(None, example=None, description="Count of pauses detected.")
    confidence_char_list: Optional[List[str]] = Field(None, example=["p", "ʤ", "s", "ə", "m"], description="List of characters with confidence levels.")
    missing_char_list: Optional[List[str]] = Field(None, example=["f", "g", "r", "ɑ"], description="List of missing characters.")
    construct_text: Optional[str] = Field(None, example="jumps", description="Constructed text based on the hypothesis.")
    tempo_classification: Optional[str] = Field(None, example="Natural", description="Tempo classification")
    words_per_minute: Optional[float] = Field(None, example=120, description="Estimated Words Per Minute")
    rate_classification: Optional[str] = Field(None, example="Natural", description="Rate classification based on estimated WPM only.")

class AudioProcessingResponse(BaseModel):
    denoised_audio_base64: str = Field(..., example="UkiGRV////wqgwbwrbw////AAAA", description="Base64 encoded denoised audio.")
    pause_count: Optional[int] = Field(..., example=2, description="Count of pauses detected.")
    avg_pause: Optional[float] = Field(None, example=0.7, description="Average pause duration in seconds.")
    smoothness_classification: Optional[str] = Field(None, description="Smoothness classification result.")
    pitch_classification: Optional[str] = Field(None, example="Natural", description="Pitch classification result (Flat, Natural, Exaggerated, or Erratic).")
    pitch_mean: Optional[float] = Field(None, example=200, description="Average pitch in Hz.")
    pitch_std: Optional[float] = Field(None, example=30, description="Standard deviation of pitch.")
    intensity_classification: Optional[str] = Field(None, example="Natural", description="Intensity classification result (Flat, Natural, Exaggerated, Erratic).")
    intensity_mean: Optional[float] = Field(None, example=65, description="Average intensity in dB.")
    intensity_std: Optional[float] = Field(None, example=8, description="Standard deviation of intensity.")
    expression_classification: Optional[str] = Field(None, description="Expression classification result.")

class ReadingComplexityRequest(BaseModel):
    text: str = Field(..., example="ಸಾಮರ್ಥ್ಯ", description="The text to analyze for reading complexity.")
    language: str = Field(..., example="kn", description="The language code (kn, te, hi).")

class ReadingComplexityResponse(BaseModel):
    total_score: float = Field(..., example=5.2, description="Total complexity score for the text.")