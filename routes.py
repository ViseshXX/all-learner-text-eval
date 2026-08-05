import base64
import io
import logging
from fastapi import APIRouter, HTTPException, Depends
import numpy as np
from pydantic import BaseModel
from utils import calculate_wpm_from_audio, classify_expression, classify_intensity, classify_rate, classify_smoothness, classify_tempo, denoise_with_rnnoise, extract_intensity_praat, get_error_arrays, get_pause_count, split_into_phonemes, processLP, extract_pitch_praat, classify_pitch, calculate_reading_complexity, get_score
from schemas import TextData, audioData, PhonemesRequest, PhonemesResponse, ErrorArraysResponse, AudioProcessingResponse, ReadingComplexityRequest, ReadingComplexityResponse
from typing import List
import jiwer
import eng_to_ipa as p

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post('/getTextMatrices', response_model=ErrorArraysResponse, summary="Compute Text Matrices", description="Computes WER, CER, insertion, deletion, substitution, confidence char list, missing char list, construct text", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Reference text must be provided."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: Error processing characters: <error_message>"}
            }
        }
    }
})
async def compute_errors(data: TextData):
    try:
        # Validate input data
        if not data.reference:
            raise HTTPException(status_code=400, detail="Reference text must be provided.")

        reference = data.reference
        hypothesis = data.hypothesis if data.hypothesis is not None else ""
        language = data.language


        # Validate language
        allowed_languages = {"en", "ta", "te", "kn", "hi", "gu", "or", "ne"}

        if language not in allowed_languages:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}. Supported languages are: {', '.join(allowed_languages)}")

        # Process character-level differences
        try:
            charOut = jiwer.process_characters(reference, hypothesis)
        except Exception as e:
            logger.error(f"Error processing characters: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing characters: {str(e)}")

        # Compute WER
        try:
            wer = jiwer.wer(reference, hypothesis)
        except Exception as e:
            logger.error(f"Error computing WER: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error computing WER: {str(e)}")

        confidence_char_list = []
        missing_char_list = []
        construct_text = ""

        if language == "en":
            try:
                confidence_char_list, missing_char_list, construct_text = processLP(reference, hypothesis)
            except Exception as e:
                logger.error(f"Error processing LP: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing LP: {str(e)}")

        # Extract error arrays
        try:
            error_arrays = get_error_arrays(charOut.alignments, reference, hypothesis)
        except Exception as e:
            logger.error(f"Error extracting error arrays: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting error arrays: {str(e)}")
        
        tempo_wpm = 0.0
        tempo_class = None
        rate_class = None
        pause_count = None
        if data.base64_string and hypothesis.strip():
            # Use your new utility function
            tempo_wpm = calculate_wpm_from_audio(hypothesis, data.base64_string)
            try:
                import base64, io
                audio_bytes = base64.b64decode(data.base64_string)
                audio_io = io.BytesIO(audio_bytes)
                p_count, avg_pause  = get_pause_count(audio_io)
                pause_count = p_count
            except Exception as e:
                logger.error(f"Error processing pause count: {str(e)}")
                pause_count = None

            # Determine if reference is a single word
            single_word = len(reference.split()) == 1
            # Classify tempo using both wpm and pause_count
            tempo_class = classify_tempo(tempo_wpm, pause_count, language, single_word)
            rate_class = classify_rate(tempo_wpm, language, single_word)

        return {
            "wer": wer,
            "cer": charOut.cer,
            "insertion": error_arrays['insertion'],
            "insertion_count": len(error_arrays['insertion']),
            "deletion": error_arrays['deletion'],
            "deletion_count": len(error_arrays['deletion']),
            "substitution": error_arrays['substitution'],
            "substitution_count": len(error_arrays['substitution']),
            "confidence_char_list": confidence_char_list,
            "missing_char_list": missing_char_list,
            "construct_text": construct_text,
            "tempo_classification": tempo_class,
            "words_per_minute": tempo_wpm,
            "rate_classification": rate_class,
            "pause_count": pause_count
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@router.post("/getPhonemes", response_model=PhonemesResponse, summary="Get Phonemes", description="Converts text into phonemes.", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Input text cannot be empty."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: Error getting phonemes: <error_message>"}
            }
        }
    }
})
async def get_phonemes(data: PhonemesRequest):
    try:
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        phonemesList = split_into_phonemes(p.convert(data.text))
        return {"phonemes": phonemesList}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting phonemes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting phonemes: {str(e)}")
    
@router.post('/audio_processing', response_model=AudioProcessingResponse, summary="Process Audio", description="Processes audio by denoising and detecting pauses.", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Base64 string of audio must be provided."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: <error_message>"}
            }
        }
    }
})
async def audio_processing(data: audioData):
    try:
        # Validate input data
        if not data.base64_string:
            raise HTTPException(status_code=400, detail="Base64 string of audio must be provided.")
        if not data.contentType:
            raise HTTPException(status_code=400, detail="Content type must be specified.")
        
        try:
            audio_data = data.base64_string
            audio_bytes = base64.b64decode(audio_data)
            audio_io = io.BytesIO(audio_bytes)
        except Exception as e:
            logger.error(f"Invalid base64 string: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 string: {str(e)}")

        pause_count = None
        avg_pause = None
        denoised_audio_base64 = ""
        pitch_classification = None
        pitch_mean = None
        pitch_std = None
        intensity_classification = None
        intensity_mean = None
        intensity_std = None
        expression_classification = None
        smoothness_classification = None

        if data.enablePauseCount:
            try:
                p_count, avg_pause = get_pause_count(audio_io)
                pause_count = p_count
                smoothness_classification = classify_smoothness(p_count, avg_pause)
                if pause_count is None:
                    logger.error("Error during pause count detection")
                    raise HTTPException(status_code=500, detail="Error during pause count detection")
            except Exception as e:
                logger.error(f"Error during pause count detection: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error during pause count detection: {str(e)}")
            # IMPORTANT: Re-seek audio_io if used again, or re-decode audio because we've consumed it
            audio_io.seek(0)

        if data.enableDenoiser:
            try:
                denoised_audio_base64 = denoise_with_rnnoise(audio_data, data.contentType)
                if denoised_audio_base64 is None:
                    logger.error("Error during audio denoising")
                    raise HTTPException(status_code=500, detail="Error during audio denoising")
            except ValueError as e:
                logger.error(f"Value error in denoise_with_rnnoise: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Value error in denoise_with_rnnoise: {str(e)}")
            except RuntimeError as e:
                logger.error(f"Runtime error in denoise_with_rnnoise: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Runtime error in denoise_with_rnnoise: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error in denoise_with_rnnoise: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Unexpected error in denoise_with_rnnoise: {str(e)}")
        else:
            # If denoiser is not enabled, just keep original audio
            denoised_audio_base64 = audio_data    
            
        if data.enable_prosody_fluency:
            try:
                denoised_audio_bytes = base64.b64decode(denoised_audio_base64)

                # If we need pitch (or expression requires pitch)
                pitch_values = extract_pitch_praat(denoised_audio_bytes)
                if pitch_values.size > 0:
                    pitch_classification = classify_pitch(pitch_values)
                    pitch_mean = float(np.round(np.mean(pitch_values), 2))
                    pitch_std = float(np.round(np.std(pitch_values), 2))
                else:
                    pitch_classification = "N/A"

                # If we need intensity (or expression requires intensity)
                intensity_values = extract_intensity_praat(denoised_audio_bytes, intensity_threshold=30)
                if intensity_values.size > 0:
                    intensity_classification = classify_intensity(intensity_values)
                    intensity_mean = float(np.round(np.mean(intensity_values), 2))
                    intensity_std = float(np.round(np.std(intensity_values), 2))
                else:
                    intensity_classification = "N/A"
                # If expression is requested, we need both pitch & intensity
                expression_classification = classify_expression(pitch_values, intensity_values)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error computing prosody and fluency: {str(e)}")

        return {
            "denoised_audio_base64": denoised_audio_base64,
            "pause_count": pause_count,
            "avg_pause": avg_pause,
            "smoothness_classification": smoothness_classification,
            "pitch_classification": pitch_classification,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "intensity_classification": intensity_classification,
            "intensity_mean": intensity_mean,
            "intensity_std": intensity_std,
            "expression_classification": expression_classification,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post('/getReadingComplexity', response_model=ReadingComplexityResponse, summary="Calculate Reading Complexity", description="Calculates reading complexity score for text in supported languages.", responses={
    400: {
        "description": "Bad Request",
        "content": {
            "application/json": {
                "example": {"detail": "Text cannot be empty."}
            }
        }
    },
    422: {
        "description": "Unprocessable Entity",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Unexpected error: <error_message>"}
            }
        }
    }
})
async def get_reading_complexity(data: ReadingComplexityRequest):
    try:
        # Validate input data
        if not data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
        # Validate language
        allowed_languages = {"kn", "te", "hi", "ne"}
        if data.language not in allowed_languages:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {data.language}. Supported languages are: {', '.join(allowed_languages)}")

        # Calculate reading complexity
        try:
            total_score, scores_data, syllable_count = calculate_reading_complexity(data.text, data.language)
        except Exception as e:
            logger.error(f"Error calculating reading complexity: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calculating reading complexity: {str(e)}")

        return {
            "total_score": total_score
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
