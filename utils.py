import base64
import io
import wave
import ffmpeg
from functools import lru_cache
import eng_to_ipa as p
from fuzzywuzzy import fuzz
import numpy as np
import soundfile as sf
import parselmouth
import regex
import configs

english_phoneme = ["b","d","f","g","h","ʤ","k","l","m","n","p","r","s","t","v","w","z","ʒ","tʃ","ʃ","θ","ð","ŋ","j","æ","eɪ","ɛ","i:","ɪ","aɪ","ɒ","oʊ","ʊ","ʌ","u:","ɔɪ","aʊ","ə","eəʳ","ɑ:","ɜ:ʳ","ɔ:","ɪəʳ","ʊəʳ","i","u","ɔ","ɑ","ɜ","e","ʧ","o","y","a", "x", "c"]
anamoly_list = {}

def denoise_with_rnnoise(audio_base64, content_type, padding_duration=0.1, time_stretch_factor=0.75):
    try:
        # Decode base64 to get the audio data
        try:
            audio_data = base64.b64decode(audio_base64)
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 string: {str(e)}")

        audio_io = io.BytesIO(audio_data)
        input_audio = audio_io.read()

        # Path to the RNNoise model
        model_path = "./audio_model/cb.rnnn"

        # Create the ffmpeg filter chain
        filter_chain = []
        if content_type.lower() == 'word':
            filter_chain.append(f'apad=pad_dur={padding_duration}')
            filter_chain.append(f'apad=pad_dur={padding_duration}')
        filter_chain.append(f'atempo={time_stretch_factor}')
        filter_chain_str = ','.join(filter_chain)

        # Apply the filters and denoise
        try:
            output, _ = (
                ffmpeg
                .input('pipe:', format='wav')
                .output('pipe:', format='wav', af=f'{filter_chain_str},arnndn=m={model_path}')
                .run(input=input_audio, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error during noise reduction with FFmpeg: {e.stderr.decode()}")

        # Convert the processed output back to base64
        try:
            denoised_audio_base64 = base64.b64encode(output).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error encoding output to base64: {str(e)}")
        
        # Clear cache to free memory
        del audio_data
        del audio_io

        return denoised_audio_base64

    except ValueError as e:
        print(f"Value error in denoise_with_rnnoise: {str(e)}")
        raise
    except RuntimeError as e:
        print(f"Runtime error in denoise_with_rnnoise: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error in denoise_with_rnnoise: {str(e)}")
        raise

def convert_to_base64(audio_data, sample_rate):
    try:
        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_data, sample_rate, format='wav')
        except Exception as e:
            raise RuntimeError(f"Error writing audio data to buffer: {str(e)}")

        buffer.seek(0)
        try:
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Error encoding buffer to base64: {str(e)}")

        return base64_audio
    except Exception as e:
        print(f"Error in convert_to_base64: {str(e)}")
        return {"error": str(e)}
       
def get_error_arrays(alignments, reference, hypothesis):
    insertion = []
    deletion = []
    substitution = []

    for chunk in alignments[0]:
        if chunk.type == 'insert':
            insertion.extend(
                list(range(chunk.hyp_start_idx, chunk.hyp_end_idx)))
        elif chunk.type == 'delete':
            deletion.extend(
                list(range(chunk.ref_start_idx, chunk.ref_end_idx)))
        elif chunk.type == 'substitute':
            refslice = slice(chunk.ref_start_idx, chunk.ref_end_idx)
            hyposlice = slice(chunk.hyp_start_idx, chunk.hyp_end_idx)

            substitution.append({
                "removed": hypothesis[hyposlice],
                "replaced": reference[refslice]
            })

    insertion_chars = [hypothesis[i] for i in insertion]
    deletion_chars = [reference[i] for i in deletion]

    return {
        'insertion': insertion_chars,
        'deletion': deletion_chars,
        'substitution': substitution, 
    }

def get_pause_count(audio_io):
    # Read the entire data from audio_io
    original_audio_data = audio_io.read()
    audio_io.seek(0)     # Rewind if you need to use audio_io again

    # Load audio samples for processing
    samples, sr = sf.read(io.BytesIO(original_audio_data), dtype="float32")
    if sr <= 0 or len(samples) == 0:
        # If something is invalid, just return 0, 0.0
        return 0, 0.0
    total_duration = len(samples) / sr
    skip_seconds = 0.75

    # If we can't skip 0.75s from start & end, there's no "middle"
    if total_duration <= 2 * skip_seconds:
        # We'll treat that as no valid portion to analyze => return zeros
        return 0, 0.0

    start_sample = int(sr * skip_seconds)
    end_sample = int(sr * (total_duration - skip_seconds))
    middle_samples = samples[start_sample:end_sample]

    # Convert float32 -> int16
    middle_int16 = (middle_samples * 32767).astype(np.int16)

    # Create in-memory WAV of that portion
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)      # assume mono
        w.setsampwidth(2)     # 16-bit
        w.setframerate(sr)
        w.writeframes(middle_int16.tobytes())
    buf.seek(0)

    # Run ffmpeg silencedetect
    process = (
        ffmpeg
        .input('pipe:0')
        .filter('silencedetect', noise='-40dB', duration=0.5)
        .output('pipe:1', format='null')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    stdout, stderr = process.communicate(input=buf.read())

    lines = stderr.decode().split('\n')
    # Count how many times we see "silence_start"
    pause_count = sum(1 for line in lines if "silence_start" in line)

    # Also gather durations from "silence_duration"
    pause_durations = []
    for line in lines:
        if "silence_duration" in line:
            try:
                parts = line.split("silence_duration:")
                if len(parts) > 1:
                    dur_str = parts[1].strip().split()[0]
                    pause_durations.append(float(dur_str))
            except:
                pass

    avg_pause = sum(pause_durations) / len(pause_durations) if pause_durations else 0.0
    avg_pause = np.round(avg_pause, 2)
    return pause_count, avg_pause

def find_closest_match(target_word, input_string):
    # Tokenize the input string into words
    words = input_string.lower().split()
    targ = target_word.lower()
    # Initialize variables to keep track of the best match
    best_match = None
    best_score = 0
    
    # Iterate through the words in the input string
    for word in words:
        similarity_score = fuzz.ratio(targ, word)
        
        # Update the best match if a higher score is found
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = word
    
    return best_match, best_score

@lru_cache(maxsize=None)
def split_into_phonemes(token):
    # Phoneme mapping for combined phonemes
    combined_phonemes = {
        "dʒ": "ʤ",
        "tʃ": "ʧ",
        "ɪəʳ": "ɪəʳ",
        "ʊəʳ": "ʊəʳ",
        "eɪʳ": "eɪ",
        "aɪ": "aɪ",
        "oʊ": "o",
        "ɔɪ": "ɔɪ",
        "aʊ": "aʊ",
        "eəʳ": "eəʳ",
        "ɑ:": "ɑ",
        "ɜ:ʳ": "ɜ:ʳ",
        "ɔ:": "ɔ:",
        "i:": "i",
    }
    
    # Set of characters to skip (stress marks, etc.)
    skip_chars = {"'", " ", "ˈ", "ˌ"}

    # Convert the english_phoneme list into a set for O(1) average-time complexity checks
    english_phoneme_set = set(english_phoneme)
    
    ph_list = []
    word_list = token.split()  # split by whitespace (space, tab, newline, etc.)

    for p in word_list:
        size = len(p)
        i = 0
        while i < size:
            if p[i] in skip_chars:
                i += 1
                continue

            # Check for combined phonemes first (3 then 2 characters long)
            if i + 3 <= size and p[i:i+3] in combined_phonemes:
                ph_list.append(combined_phonemes[p[i:i+3]])
                i += 3
            elif i + 2 <= size and p[i:i+2] in combined_phonemes:
                ph_list.append(combined_phonemes[p[i:i+2]])
                i += 2
            elif i + 1 <= size and p[i:i+1] in english_phoneme_set:
                ph_list.append(p[i:i+1])
                i += 1
            else:
                # Log an anomaly if the character isn't recognized
                ph_list.append(p[i])
                if p[i] not in anamoly_list:
                    anamoly_list[p[i]] = 1
                else:
                    anamoly_list[p[i]] += 1
                i += 1

    return ph_list

def identify_missing_tokens(orig_text, resp_text):
    # Splitting text into words
    orig_word_list = orig_text.lower().split()
    resp_word_list = resp_text.lower().split()
    
    # Initialize lists and dictionaries
    construct_word_list = []
    missing_word_list = []
    orig_phoneme_list = []
    construct_phoneme_list = []
    missing_phoneme_list = []
    construct_text = []
    
    # Precompute phonemes for response words for quick lookup
    resp_phonemes = {word: p.convert(word).replace('*', '') for word in resp_word_list}
    for word in orig_word_list:
        # Precompute original word phonemes
        p_word = p.convert(word)
        
        # Find closest match based on precomputed phonemes to avoid redundant calculations
        closest_match, similarity_score = find_closest_match(word, resp_text)
        
        # Check similarity and categorize word
        if similarity_score > 80:
            construct_word_list.append(closest_match)
            p_closest_match = resp_phonemes[closest_match]
            construct_phoneme_list.append(split_into_phonemes(p_closest_match))
            construct_text.append(closest_match)
        else:
            missing_word_list.append(word)
            p_word_phonemes = split_into_phonemes(p_word)
            missing_phoneme_list.append(p_word_phonemes)
        
        # Store original phonemes for each word
        orig_phoneme_list.append(split_into_phonemes(p_word))

    # Convert list of words to a single string
    construct_text = ' '.join(construct_text)

    # Efficiently deduplicate and flatten phoneme lists
    #orig_flatList = set(phoneme for sublist in orig_phoneme_list for phoneme in sublist)
    missing_flatList = set(phoneme for sublist in missing_phoneme_list for phoneme in sublist)
    construct_flatList = set(phoneme for sublist in construct_phoneme_list for phoneme in sublist)

    return list(construct_flatList), list(missing_flatList) ,construct_text

def processLP(orig_text, resp_text):
    cons_list, miss_list, construct_text = identify_missing_tokens(orig_text, resp_text)

    #remove phonemes from miss_list which are in cons_list, ?but add those phonemes a count of could be issu
    # phonemes in constructed list are familiar ones
    # phonemes that are in miss_list and not in cons_list are the unfamiliar ones
    unfamiliar_list = []
    for c in miss_list:
        if c not in cons_list:
            unfamiliar_list.append(c)
    #function to calculate wer cer, substitutions, deletions and insertions, silence, repetitions
    #insert into DB the LearnerProfile vector
    return cons_list, miss_list,construct_text

def read_wave_bytes(wav_data: bytes):
    with io.BytesIO(wav_data) as mem:
        with wave.open(mem, "rb") as f:
            num_channels = f.getnchannels()
            sampwidth = f.getsampwidth()
            assert sampwidth == 2, f"Expected 16-bit, got {sampwidth * 8}-bit"
            sample_rate = f.getframerate()
            num_frames = f.getnframes()
            data = f.readframes(num_frames)

    samples_int16 = np.frombuffer(data, dtype=np.int16)
    if num_channels > 1:
        samples_int16 = samples_int16.reshape(-1, num_channels)
        samples_int16 = samples_int16.mean(axis=1).astype(np.int16)

    samples_float32 = samples_int16.astype(np.float32) / 32768.0
    return samples_float32, sample_rate

def extract_pitch_praat(wav_data: bytes, pitch_floor=75, pitch_ceiling=500):
    try:
        samples, sr = read_wave_bytes(wav_data)
        sound = parselmouth.Sound(samples, sr)
        pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]
        return pitch_values
    except Exception as e:
        raise RuntimeError(f"Error extracting pitch with parselmouth: {e}")

def classify_pitch(pitch_values):
    if pitch_values.size == 0:
        return "N/A"
    mean_pitch = np.round(np.mean(pitch_values), 2)
    std_pitch = np.round(np.std(pitch_values), 2)

    if 120 <= mean_pitch <= 400:
        if std_pitch < 15:
            return "Flat"
        elif std_pitch <= 70:
            return "Natural"
        elif std_pitch <= 100:
            return "Exaggerated"
        else:
            return "Erratic"
    else:
        # If out of typical range, treat as either "Erratic" or "Exaggerated"
        if std_pitch > 50:
            return "Erratic"
        else:
            return "Exaggerated"
        
def extract_intensity_praat(wav_data: bytes, intensity_threshold=30):
    try:
        samples, sr = read_wave_bytes(wav_data)
        sound = parselmouth.Sound(samples, sr)
        intensity_obj = sound.to_intensity()
        # intensity_obj.values has shape (1, n_frames).
        # Flatten to get a 1D array of intensities
        intensity_values = intensity_obj.values.flatten()

        # Filter out intensities below the threshold
        intensity_values = intensity_values[intensity_values >= intensity_threshold]
        return intensity_values
    except Exception as e:
        raise RuntimeError(f"Error extracting intensity with parselmouth: {e}")


def classify_intensity(intensity_values):
    if intensity_values.size == 0:
        return "N/A"

    mean_intensity = np.round(np.mean(intensity_values), 2)
    std_intensity = np.round(np.std(intensity_values), 2)
    
    if 40 <= mean_intensity <= 80:
        if std_intensity < 6:
            return "Flat"
        elif std_intensity <= 20:
            return "Natural"
        elif std_intensity <= 25:
            return "Exaggerated"
        else:
            return "Erratic"
    else:
        # Outside typical range -> call it Exaggerated or Erratic
        if std_intensity > 10:
            return "Erratic"
        else:
            return "Exaggerated"
        
def classify_expression(pitch_values, intensity_values):

    if pitch_values.size == 0 or intensity_values.size == 0:
        return "N/A"

    # Helper to convert mean / std into a 1‑4 score
    def _score(mean, std, cfg):
        """cfg = {'mean': (lo, hi), 'std': [(lo, hi, pts), …]}"""
        if not (cfg['mean'][0] <= mean <= cfg['mean'][1]):
            # Outside the "typical mean" window ⇒ use the fallback rule
            return 1 if std > cfg['fallback_std_gt'] else 2

        for lo, hi, pts in cfg['std']:
            if lo <= std <= hi:
                return pts
        # If no band matched (shouldn’t happen), default to worst score
        return 1

    # Threshold tables for pitch & intensity
    PITCH_CFG = {
        'mean': (120, 400),
        'std': [
            (15, 70, 4),
            (0, 15, 3),
            (70, 100, 2),
            (100, float('inf'), 1),
        ],
        'fallback_std_gt': 50,   # if mean outside range
    }

    INT_CFG = {
        'mean': (40, 80),
        'std': [
            (6, 20, 4),
            (20, 25, 3),
            (25, float('inf'), 2),
            (0, 6, 1),
        ],
        'fallback_std_gt': 10,
    }

    # Compute scores 
    pitch_score     = _score(np.mean(pitch_values),     np.std(pitch_values),     PITCH_CFG)
    intensity_score = _score(np.mean(intensity_values), np.std(intensity_values), INT_CFG)

    avg_score = (pitch_score + intensity_score) // 2

    # Map average score to label 
    return {
        4: "Fluent",
        3: "Moderately Fluent",
        2: "Disfluent",
    }.get(avg_score, "Very Disfluent")
    
def classify_smoothness(pause_count, avg_pause):
    # Handle None values
    if pause_count is None or avg_pause is None:
        return None
    
    #  Count-based score
    if pause_count <= 2:
        count_score = 4
    elif pause_count <= 4:
        count_score = 3
    elif pause_count <= 6:
        count_score = 2
    else:
        count_score = 1

    #  Duration-based score
    if avg_pause <= 0.6:
        duration_score = 4
    elif avg_pause <= 1.0:
        duration_score = 3
    elif avg_pause <= 1.5:
        duration_score = 2
    else:
        duration_score = 1

    overall_score = (count_score + duration_score) // 2

    if overall_score == 4:
        return "Fluent"
    elif overall_score == 3:
        return "Moderately Fluent"
    elif overall_score == 2:
        return "Disfluent"
    else:
        return "Very Disfluent"

def calculate_wpm_from_audio(hypothesis_text: str, base64_audio: str) -> float:

    if not base64_audio:
        return 0.0
    if not hypothesis_text:
        return 0.0

    try:
        #  Decode base64
        audio_data = base64.b64decode(base64_audio)
        audio_io = io.BytesIO(audio_data)
        #  Read with soundfile
        samples, sr = sf.read(audio_io, dtype="float32")
        duration_seconds = len(samples) / sr if sr else 0.0
        duration_minutes = duration_seconds / 60.0 if duration_seconds > 0 else 0

        #  Word count from hypothesis
        word_count = len(hypothesis_text.strip().split()) if hypothesis_text.strip() else 0

        #  WPM
        if duration_minutes > 0 and word_count > 0:
            wpm = word_count / duration_minutes
        else:
            wpm = 0.0
        return np.round(wpm, 2)
    except Exception as e:
        print(f"Error calculating WPM: {str(e)}")
        return 0.0
    
def compute_wpm_score(estimated_wpm: float, language: str, single_word: bool) -> int:

    language = language.lower()

    # Each tuple is (lower_bound_inclusive, upper_bound_exclusive, score)
    SINGLE_WORD_BANDS = [
        (0,   10,   1),   # Erratic
        (10,  20,   2),   # Exaggerated
        (20,  30,   3),   # Flat
        (30,  110,  4),   # Natural
        (110, 120,  2),   # Exaggerated
        (120, float("inf"), 1)  # Erratic
    ]

    INDIC_BANDS = [
        (0,   15,   1),
        (15,  30,   2),
        (30,  50,   3),
        (50,  140,  4),
        (140, 180,  2),
        (180, float("inf"), 1)
    ]

    DEFAULT_BANDS = [
        (0,   40,   1),
        (40,  60,   2),
        (60,  100,  3),
        (100, 180,  4),
        (180, 240,  2),
        (240, float("inf"), 1)
    ]

    if single_word:
        bands = SINGLE_WORD_BANDS
    else:
        bands = DEFAULT_BANDS if language == "en" else INDIC_BANDS

    # Return the first matching score 
    for low, high, score in bands:
        if low <= estimated_wpm < high:
            return score

    # Fallback (should never be hit with the tables above)
    return 1

def classify_tempo(estimated_wpm: float, pause_count: int, language: str, single_word: bool = False) -> str:
    # Compute the wpm score using the helper function.
    wpm_score = compute_wpm_score(estimated_wpm, language, single_word)

    # Determine a pause score.
    if pause_count <= 2:
        pause_score = 4
    elif pause_count <= 4:
        pause_score = 3
    elif pause_count <= 6:
        pause_score = 2
    else:
        pause_score = 1

    # Calculate the weighted score.
    weighted_score = (2 * wpm_score + pause_score) // 3

    # Map the weighted score to a classification string.
    if weighted_score == 4:
        return "Natural"
    elif weighted_score == 3:
        return "Flat"
    elif weighted_score == 2:
        return "Exaggerated"
    else:
        return "Erratic"

def classify_rate(estimated_wpm: float, language: str, single_word: bool = False) -> str:


    # Each tuple is (lower_bound_inclusive, upper_bound_exclusive, label)
    SINGLE_BANDS = [
        (0,   10,   "Very Disfluent"),
        (10,  20,   "Disfluent"),
        (20,  30,   "Moderately Fluent"),
        (30,  110,  "Fluent"),
        (110, 120,  "Disfluent"),
        (120, float("inf"), "Very Disfluent"),
    ]

    INDIC_BANDS = [
        (0,   15,   "Very Disfluent"),
        (15,  30,   "Disfluent"),
        (30,  50,   "Moderately Fluent"),
        (50,  140,  "Fluent"),
        (140, 180,  "Disfluent"),
        (180, float("inf"), "Very Disfluent"),
    ]

    DEFAULT_BANDS = [
        (0,   40,   "Very Disfluent"),
        (40,  60,   "Disfluent"),
        (60,  100,  "Moderately Fluent"),
        (100, 180,  "Fluent"),
        (180, 240,  "Disfluent"),
        (240, float("inf"), "Very Disfluent"),
    ]

    # Select the correct band table 
    language = language.lower()
    bands = (
        SINGLE_BANDS if single_word
        else DEFAULT_BANDS if language == "en"
        else INDIC_BANDS
    )

    #  Return the first matching label 
    for low, high, label in bands:
        if low <= estimated_wpm < high:
            return label

    # Fallback (should never fire with the tables above)
    return "Very Disfluent"

# Reading Complexity Functions
def find_syllables(word, language):
    """Find syllables for a given word based on language configuration."""
    config = configs.language_data
    letter = config[language]["regex"]["letter"]
    trailing_letter = config[language]["regex"]["trailing_letter"]
    control = config[language]["regex"]["control"]
    regex_exp = rf'{letter}(?:{control}{letter}|{trailing_letter})*'
    syllables = regex.findall(regex_exp, word)
    return syllables

def is_samyukta(syllable, language):
    """Check if a syllable is samyutkashara and return included consonants."""
    config = configs.language_data
    virama = config[language]["virama"]
    consonants = []
    for i, char in enumerate(syllable):
        if (i > 0 and syllable[i - 1] == virama) or (i < len(syllable) - 1 and syllable[i + 1] == virama):
            consonants.append(char)
    return consonants

def replace_nukta_chars(word, language):
    """Replace nukta characters in Hindi text."""
    config = configs.language_data
    if language != 'hi' or 'nukta' not in config[language]:
        return word
    
    nukta = config[language]['nukta']
    char_replace = config[language]['Nukta_char_replace']
    chars = list(word)
    i = 1
    while i < len(chars):
        if chars[i] == nukta:
            base = chars[i - 1]
            if base in char_replace:
                # Replace base + nukta with nukta form
                chars[i - 1] = char_replace[base]
                del chars[i]  # Remove nukta char
                continue  # Re-check the same index (since we removed one char)
        i += 1
    return ''.join(chars)

def add_arkavattu_score(samyukta_arr, language):
    """Check for arkavattu and return score."""
    config = configs.language_data
    if "arkavattu" in config[language] and samyukta_arr and samyukta_arr[0] == config[language]["arkavattu"]:
        return 1.5
    return 0

def get_score(word, language):
    """Calculate complexity score for a single word."""
    config = configs.language_data
    scores = config[language]["score"]
    addedscores = []
    word = word.replace(" ", "")
    
    if ("nukta" in config[language] and config[language]["nukta"] in word):
        word = replace_nukta_chars(word, language)
    
    syllables = find_syllables(word, language)
    score = 0
    syllable_count_weightage = (len(syllables) - 1) * 0.5
    score += syllable_count_weightage
    addedscores.append(syllable_count_weightage)
    
    for syllable in syllables:
        similarities = set()
        resp = is_samyukta(syllable, language)
        
        # Check for arkavattu in language and add the score of it.
        if ("arkavattu" in config[language] and config[language]["arkavattu"] in syllable and len(resp) > 0):
            arka_score = add_arkavattu_score(resp, language)
            if arka_score > 0:
                addedscores.append(1.5)
                score = score + arka_score
        
        # Add the Samyutakshra Weightage
        if len(resp) > 0:
            if len(resp) == 2:  # If there are two consonants
                if resp[0] == resp[1]:  # If they are the same, the Weight_base of that consonant is added. along with additional weight of 1
                    score += scores[resp[0]]['Weight_base'] + 1
                    addedscores.append(scores[resp[0]]['Weight_base'] + 1)
                else:  # If they are different, the Weight_base of both consonants is added. along with additional weight of 2
                    score += scores[resp[0]]['Weight_base'] + scores[resp[1]]['Weight_base'] + 2
                    addedscores.append(scores[resp[0]]['Weight_base'] + scores[resp[1]]['Weight_base'] + 2)
            
            if len(resp) > 2:  # If there are more than two consonants, the Weight_base of 3 consonants is added along with additional 3 points are added.
                score += scores[resp[0]]['Weight_base'] + scores[resp[1]]['Weight_base'] + scores[resp[2]]['Weight_base'] + 3
                addedscores.append(scores[resp[0]]['Weight_base'] + scores[resp[1]]['Weight_base'] + scores[resp[2]]['Weight_base'] + 3)
        
        for char in syllable:
            if char in scores:
                score += scores[char]['Weight']  # Add the individual Char weightage.
                addedscores.append(scores[char]['Weight'])
                
                if scores[char]['similar'] > 0:
                    if len(similarities) == 0:
                        score += 0.6  # Add the similar score if applicable.
                        addedscores.append(0.6)
                        similarities.add(char)
    
    score = round(score, 2)
    return score, addedscores, len(syllables)

def calculate_reading_complexity(content, language):
    """Calculate reading complexity score for entire content."""
    content = regex.sub(r'[^\w\s]', '', content)
    split_content = content.split()
    final_scores = 0
    scores_data = []
    syllb_count = 0
    
    for word in split_content:
        output = get_score(word, language)
        final_scores += output[0]
        scores_data.append(output[1])
        syllb_count += output[2]
    final_scores = round(final_scores, 2)

    return final_scores, scores_data, syllb_count