# app/pipeline.py
from pathlib import Path
import subprocess, json
import tempfile
import asyncio
from typing import List, Optional, Dict, Any, Union
import re
from .config import MODEL_PATH, WHISPER_PATH, validate_paths
from .utils import simulate_delay, analyze_segment_types, AnalysisSegment

# Add imports for the new functions
import num2words
import contractions
from jiwer import WordOutput, process_words, visualize_alignment, wer, mer, wil, wip, cer
import Levenshtein
import cmudict
from g2p_en import G2p
from phonecodes import phonecodes

# ---------------------------------------------------------------------------
# Phoneme analysis
# ---------------------------------------------------------------------------

# Initialize g2p and cmudict
g2p = G2p()
cmu_dict = cmudict.dict()

def get_phonemes_arpabet(word: str) -> List[str]:
    """
    Get the phonemic representation of a word in ARPABET format.
    Uses cmudict first, then g2p-en as a fallback.
    """
    word = word.lower().strip()
    if not word:
        return []
    
    # 1. Check cmudict
    if word in cmu_dict:
        # Return the first pronunciation variant
        return cmu_dict[word][0]
    
    # 2. Fallback to g2p
    try:
        return g2p(word)
    except:
        # If g2p fails, return an empty list
        return []

def get_phonemes_ipa(word: str) -> List[str]:
    """
    Get the phonemic representation of a word in IPA format.
    Uses cmudict first, then g2p-en as a fallback, then converts to IPA.
    """
    # Get ARPABET phonemes first
    arpabet_phonemes = get_phonemes_arpabet(word)
    
    if not arpabet_phonemes:
        return []
    
    try:
        # Convert ARPABET to IPA using phonecodes
        arpabet_string = " ".join(arpabet_phonemes)
        ipa_string = phonecodes.convert(arpabet_string, "arpabet", "ipa")
        # Split the IPA string back into individual phonemes
        return ipa_string.split()
    except:
        # If conversion fails, return empty list
        return []

def get_phonemes(word: str, format: str = "ipa") -> List[str]:
    """
    Get the phonemic representation of a word.
    
    Parameters
    ----------
    word : str
        The word to get phonemes for
    format : str, optional
        The phoneme format to return. Either "ipa" or "arpabet". Default is "ipa".
        
    Returns
    -------
    List[str]
        List of phonemes in the specified format
    """
    if format.lower() == "arpabet":
        return get_phonemes_arpabet(word)
    elif format.lower() == "ipa":
        return get_phonemes_ipa(word)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'ipa' or 'arpabet'.")

def calculate_phonemic_analysis(alignments_fda2: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]], intelligibility_type: str) -> Dict[str, Any]:
    """
    Calculate phonemic analysis from alignment data.
    
    Parameters
    ----------
    alignments_fda2 : Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
        Alignment data from calculate_metrics function
    intelligibility_type : str
        Either "words" or "sentences"
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing phonemic analysis results
    """
    correctly_transcribed_phonemes = []
    incorrectly_transcribed_phonemes = []
    phoneme_error_mappings = {}  # ref_phoneme -> {hyp_phoneme: count}
    
    word_pairs = []
    
    if intelligibility_type == "words":
        # For words: alignments_fda2 is List[List[Dict]]
        for chunk in alignments_fda2:
            for alignment in chunk:
                ref_word = alignment.get('reference_words', '').strip()
                hyp_word = alignment.get('hypothesis_words', '').strip()
                if ref_word:  # Only process if there's a reference word
                    word_pairs.append((ref_word, hyp_word))
    
    elif intelligibility_type == "sentences":
        # For sentences: alignments_fda2 is List[Dict]
        for sentence_alignment in alignments_fda2:
            ref_sentence = sentence_alignment.get('reference', '').strip()
            hyp_sentence = sentence_alignment.get('hypothesis', '').strip()
            
            # Split sentences into words and pair them up using simple alignment
            ref_words = ref_sentence.split()
            hyp_words = hyp_sentence.split()
            
            # Simple alignment: pair words by position, handle mismatches
            max_len = max(len(ref_words), len(hyp_words))
            for i in range(max_len):
                ref_word = ref_words[i] if i < len(ref_words) else ''
                hyp_word = hyp_words[i] if i < len(hyp_words) else ''
                if ref_word:  # Only process if there's a reference word
                    word_pairs.append((ref_word, hyp_word))
    
    # Analyze each word pair for phoneme errors
    for ref_word, hyp_word in word_pairs:
        if not ref_word:
            continue
            
        # Get phonemes for both words
        ref_phonemes = get_phonemes(ref_word)
        hyp_phonemes = get_phonemes(hyp_word) if hyp_word else []
        
        if not ref_phonemes:
            continue
            
        # Compare phonemes using simple alignment
        # For simplicity, we'll use position-based alignment
        correctly_transcribed = []
        incorrectly_transcribed = []
        
        max_phoneme_len = max(len(ref_phonemes), len(hyp_phonemes))
        
        for i in range(len(ref_phonemes)):
            ref_phoneme = ref_phonemes[i]
            hyp_phoneme = hyp_phonemes[i] if i < len(hyp_phonemes) else None
            
            if hyp_phoneme and ref_phoneme == hyp_phoneme:
                # Correct transcription
                correctly_transcribed.append(ref_phoneme)
            else:
                # Incorrect transcription
                incorrectly_transcribed.append(ref_phoneme)
                
                # Track error mapping
                if ref_phoneme not in phoneme_error_mappings:
                    phoneme_error_mappings[ref_phoneme] = {}
                
                error_target = hyp_phoneme if hyp_phoneme else "DELETED"
                if error_target not in phoneme_error_mappings[ref_phoneme]:
                    phoneme_error_mappings[ref_phoneme][error_target] = 0
                phoneme_error_mappings[ref_phoneme][error_target] += 1
        
        correctly_transcribed_phonemes.extend(correctly_transcribed)
        incorrectly_transcribed_phonemes.extend(incorrectly_transcribed)
    
    # Calculate summary statistics
    total_phonemes = len(correctly_transcribed_phonemes) + len(incorrectly_transcribed_phonemes)
    phoneme_accuracy = len(correctly_transcribed_phonemes) / total_phonemes if total_phonemes > 0 else 0.0
    
    return {
        "total_phonemes": total_phonemes,
        "correctly_transcribed_phonemes": correctly_transcribed_phonemes,
        "incorrectly_transcribed_phonemes": incorrectly_transcribed_phonemes,
        "phoneme_error_mappings": phoneme_error_mappings,
        "phoneme_accuracy": phoneme_accuracy,
        "phoneme_error_count": len(incorrectly_transcribed_phonemes),
        "unique_error_phonemes": list(set(incorrectly_transcribed_phonemes)),
        "unique_correct_phonemes": list(set(correctly_transcribed_phonemes))
    }

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(i) for i in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj

async def get_audio_duration(file_path: Path) -> float:
    """Get the duration of an audio/video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            print(f"Error getting duration for {file_path}: {stderr.decode()}")
            return 0.0
        return float(stdout.decode().strip())
    except (ValueError) as e:
        print(f"Error parsing duration for {file_path}: {e}")
        return 0.0

# ---------------------------------------------------------------------------
# Audio trimming function 
# ---------------------------------------------------------------------------

def _run_ffmpeg(cmd: List[str]) -> None:
    """Run an ffmpeg command and raise if it fails."""
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode()) from e

async def trim_audio(input_file: Path, start_time: float, end_time: float, 
                    output_file: Path, status_callback=None) -> Path:
    """Trim an audio file to the specified start and end times.
    
    Parameters
    ----------
    input_file : Path
        Input audio file to trim
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    output_file : Path
        Output file path
    status_callback : Optional[Callable]
        Callback function for status updates
        
    Returns
    -------
    Path
        Path to the trimmed audio file
    """
    if status_callback:
        status_callback("converting", f"Trimming audio from {start_time}s to {end_time}s...")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(input_file),
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:a", "pcm_s16le",  # 16-bit PCM audio
        "-y",  # Overwrite output file if it exists
        str(output_file)
    ]
    
    try:
        await asyncio.to_thread(_run_ffmpeg, cmd)
        
        if status_callback:
            status_callback("trimmed", f"Audio trimmed successfully to {output_file}")
            
        return output_file
        
    except Exception as e:
        if status_callback:
            status_callback("error", f"Audio trimming failed: {str(e)}")
        raise

# ---------------------------------------------------------------------------
# Text normalization functions (copied from normalization.py)
# ---------------------------------------------------------------------------

def remove_punctuation(text: str) -> str:
    """Remove punctuation from text."""
    return re.sub(r'[^\w\s\']', ' ', text)

def expand_number(match):
    """Convert a number to words using num2words."""
    try:
        num = match.group(0)
        # Handle dollar amounts
        if num.startswith('$'):
            amount = float(num[1:].replace(',', ''))
            if amount.is_integer():
                return num2words.num2words(int(amount), to='currency', currency='USD')
            else:
                return num2words.num2words(amount, to='currency', currency='USD')
        # Handle percentages
        elif num.endswith('%'):
            amount = float(num[:-1].replace(',', ''))
            return num2words.num2words(amount) + " percent"
        # Handle regular numbers
        else:
            num_clean = num.replace(',', '')
            if '.' in num_clean:
                return num2words.num2words(float(num_clean))
            else:
                return num2words.num2words(int(num_clean))
    except (ValueError, TypeError):
        return match.group(0)

def normalize_text(text: str, 
                  lowercase: bool = True,
                  remove_punct: bool = True,
                  expand_numbers: bool = True,
                  expand_contractions: bool = True) -> str:
    """Normalize text for ASR evaluation."""
    # Process the text
    if lowercase:
        text = text.lower()
    
    # Handle contractions 
    if expand_contractions:
        text = contractions.fix(text)
    
    if remove_punct:
        text = remove_punctuation(text)
    
    # Use num2words to expand numbers
    if expand_numbers:
        # Match currency, percentages, and numbers (including decimals)
        number_pattern = r'\$\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?%|\d+(?:,\d+)*(?:\.\d+)?'
        text = re.sub(number_pattern, expand_number, text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ---------------------------------------------------------------------------
# Edit distance calculation functions (copied from edit_distance.py)
# ---------------------------------------------------------------------------

def _serialize_alignment(output: WordOutput) -> List[List[Dict[str, Any]]]:
    """Convert jiwer alignment output to a JSON-serializable format."""
    serializable_alignments = []
    for chunk_idx, alignment_chunk_list in enumerate(output.alignments):
        chunk_data_list = []
        ref_word_list = output.references[chunk_idx]
        hyp_word_list = output.hypotheses[chunk_idx]
        
        for i, alignment in enumerate(alignment_chunk_list):
            ref_words = ref_word_list[alignment.ref_start_idx:alignment.ref_end_idx]
            hyp_words = hyp_word_list[alignment.hyp_start_idx:alignment.hyp_end_idx]
            
            chunk_data_list.append({
                'index': i,
                'type': alignment.type,
                'ref_words': ref_words,
                'hyp_words': hyp_words,
                'ref_text': ' '.join(ref_words) if ref_words else '*',
                'hyp_text': ' '.join(hyp_words) if hyp_words else '*',
            })
        serializable_alignments.append(chunk_data_list)
    return serializable_alignments

def _serialize_fda2_alignment(output: WordOutput) -> List[List[Dict[str, Any]]]:
    alignment_fda2 = []
    for chunk_idx, alignment_chunk in enumerate(output.alignments):
        chunk_data = []
        ref = output.references[chunk_idx]
        hyp = output.hypotheses[chunk_idx]
        
        for i, alignment in enumerate(alignment_chunk):
            if alignment.type != 'insert':
                nb_ref_words = alignment.ref_end_idx - alignment.ref_start_idx
                for j in range(nb_ref_words):
                    # Get the actual words from the reference and hypothesis using the indices
                    ref_words = ref[alignment.ref_start_idx + j]
                    hyp_words = hyp[alignment.hyp_start_idx + j] if alignment.type != 'delete' else ""
                 
                    chunk_data.append({
                        'type': alignment.type,
                        'cer': cer(ref_words, hyp_words),
                        'wer': wer(ref_words, hyp_words),
                        'reference_words': ref_words,
                        'hypothesis_words': hyp_words,
                    })

        alignment_fda2.append(chunk_data)
    return alignment_fda2


def display_alignment(
    serialized_alignment: List[List[Dict[str, Any]]],
    show_match: bool = True,
    show_substitution: bool = True,
    show_deletion: bool = True,
    show_insertion: bool = True,
) -> str:
    """
    Generate a human-readable alignment visualization from serialized jiwer output.

    Args:
        serialized_alignment: The serialized alignment data.
        show_match: Whether to include 'equal' segments.
        show_substitution: Whether to include 'substitute' segments.
        show_deletion: Whether to include 'delete' segments.
        show_insertion: Whether to include 'insert' segments.

    Returns:
        A formatted string showing the REF, HYP, and evaluation lines.
    """
    output_lines = []
    for i, chunk in enumerate(serialized_alignment):
        ref_line_parts = []
        hyp_line_parts = []
        eval_line_parts = []

        for segment in chunk:
            ref_words = segment.get('ref_words', [])
            hyp_words = segment.get('hyp_words', [])
            seg_type = segment['type']

            if seg_type == 'equal' and show_match:
                for word in ref_words:
                    ref_line_parts.append(word)
                    hyp_line_parts.append(word)
                    eval_line_parts.append(' ' * len(word))
            
            elif seg_type == 'substitute' and show_substitution:
                max_len = max(len(ref_words), len(hyp_words))
                for i in range(max_len):
                    ref_word = ref_words[i] if i < len(ref_words) else ''
                    hyp_word = hyp_words[i] if i < len(hyp_words) else ''
                    width = max(len(ref_word), len(hyp_word))
                    
                    ref_line_parts.append(ref_word.ljust(width))
                    hyp_line_parts.append(hyp_word.ljust(width))
                    eval_line_parts.append('S'.center(width))

            elif seg_type == 'delete' and show_deletion:
                for word in ref_words:
                    ref_line_parts.append(word)
                    hyp_line_parts.append('*' * len(word))
                    eval_line_parts.append('D'.center(len(word)))

            elif seg_type == 'insert' and show_insertion:
                for word in hyp_words:
                    ref_line_parts.append('*' * len(word))
                    hyp_line_parts.append(word)
                    eval_line_parts.append('I'.center(len(word)))
        
        # Only add sentence if it has content after filtering
        if not ref_line_parts:
            continue

        ref_line = "REF: " + " ".join(ref_line_parts)
        hyp_line = "HYP: " + " ".join(hyp_line_parts)
        eval_line = "     " + " ".join(eval_line_parts)
        
        output_lines.append(f"=== SENTENCE {i+1} ===\n\n{ref_line}\n{hyp_line}\n{eval_line}\n")

    return "\n".join(output_lines)


def adjust_sentence_boundaries_for_insertions(sentence_boundaries, alignment_chunks):
    """
    Adjust sentence boundaries to account for insertions and deletions in the hypothesis.
    
    Parameters
    ----------
    sentence_boundaries : List[Tuple[int, int]]
        Original sentence boundaries as (start_idx, end_idx) tuples
    alignment_chunks : List[AlignmentChunk]
        Global alignment chunks from jiwer
        
    Returns
    -------
    List[Tuple[int, int]]
        Adjusted sentence boundaries that include insertions and deletions
        
    Rules:
    - If insertion/deletion is between sentences, add to previous sentence
    - If insertion/deletion is in middle of sentence, add to that sentence  
    - If insertion/deletion is at end of last sentence, add to last sentence
    """
    # Find all insertions and deletions and map them to sentences
    insertions_per_sentence = [0] * len(sentence_boundaries)
    deletions_per_sentence = [0] * len(sentence_boundaries)
    
    for chunk in alignment_chunks:
        if chunk.type in ['insert', 'delete']:
            # For insertions, use ref_start_idx; for deletions, use ref_start_idx as well
            ref_pos = chunk.ref_start_idx
            
            if chunk.type == 'insert':
                num_changes = chunk.hyp_end_idx - chunk.hyp_start_idx
            else:  # delete
                num_changes = chunk.ref_end_idx - chunk.ref_start_idx
            
            # Find which sentence this change should belong to based on original boundaries
            target_sentence_idx = None
            
            # Check if change is within any sentence
            for i, (start, end) in enumerate(sentence_boundaries):
                if start <= ref_pos < end:
                    target_sentence_idx = i
                    break
                elif ref_pos == end:
                    # Change is exactly at the end of a sentence
                    # Check if this is also the start of the next sentence
                    if i + 1 < len(sentence_boundaries) and sentence_boundaries[i + 1][0] == ref_pos:
                        # Between sentences - add to previous sentence
                        target_sentence_idx = i
                    else:
                        # At end of sentence - add to this sentence
                        target_sentence_idx = i
                    break
            
            # If change is beyond all sentences, add to last sentence
            if target_sentence_idx is None:
                target_sentence_idx = len(sentence_boundaries) - 1
            
            # Add changes to the target sentence
            if chunk.type == 'insert':
                insertions_per_sentence[target_sentence_idx] += num_changes
            else:  # delete
                deletions_per_sentence[target_sentence_idx] += num_changes
    
    # Apply adjustments to boundaries
    adjusted_boundaries = []
    cumulative_net_changes = 0
    
    for i, (start, end) in enumerate(sentence_boundaries):
        # Calculate net change for this sentence (insertions - deletions)
        net_change = insertions_per_sentence[i] - deletions_per_sentence[i]
        
        # Shift start and end by cumulative net changes from previous sentences
        adjusted_start = start + cumulative_net_changes
        adjusted_end = end + cumulative_net_changes + net_change
        
        adjusted_boundaries.append((adjusted_start, adjusted_end))
        cumulative_net_changes += net_change
    
    return adjusted_boundaries


def calculate_metrics(truth: Union[str, List[str]], hypothesis: str) -> Dict[str, Any]:
    """Calculate various edit distance metrics between truth and hypothesis texts."""
    # todo :assert that the hypothesis is a string and not a list of strings
    
    if isinstance(truth, str):
        # Handle simple case: single string comparison
        output = process_words(truth, hypothesis)
        alignments_fda2 = _serialize_fda2_alignment(output)
        
        # Calculate phonemic analysis
        phonemic_analysis = calculate_phonemic_analysis(alignments_fda2, "words")
        
        metrics = {
            'wer': min(output.wer, 1.0),
            'visualize_alignment': visualize_alignment(output),
            'alignments_fda2': alignments_fda2,
            'phonemic_analysis': phonemic_analysis
        }
        cer_value = Levenshtein.distance(truth, hypothesis) / len(truth) if len(truth) > 0 else 1.0
        metrics['cer'] = min(cer_value, 1.0)
        return metrics

    # Handle complex case: list of sentences vs. single hypothesis string
    reference_sentences = truth
    
    # 1. Create a single reference string and get sentence boundaries
    reference_words = []
    sentence_boundaries = []
    current_word_index = 0
    for sentence in reference_sentences:
        words = sentence.split()
        # extend add the values (words) to the reference_words list
        reference_words.extend(words)
        sentence_boundaries.append((current_word_index, current_word_index + len(words)))
        current_word_index += len(words)

    long_reference = " ".join(reference_sentences)

    # 2. Get the global alignment
    global_output = process_words(long_reference, hypothesis)
    print("global_output", visualize_alignment(global_output))
    print("global_output", global_output.alignments[0])

    # 2.5. Adjust sentence boundaries to account for insertions
    adjusted_sentence_boundaries = adjust_sentence_boundaries_for_insertions(
        sentence_boundaries, global_output.alignments[0]
    )
    print("\nOriginal boundaries:", sentence_boundaries)
    print("Adjusted boundaries:", adjusted_sentence_boundaries)
    
    # 3. Calculate per-sentence metrics by segmenting the global alignment
    sentence_results = []
    global_hyp_words = global_output.hypotheses[0]

    # for each initial reference sentence
    for i, (start_idx, end_idx) in enumerate(sentence_boundaries):
        (adjusted_start_idx, adjusted_end_idx) = adjusted_sentence_boundaries[i]
        sentence_ref_text = reference_sentences[i]
        
        # Alignement chunks for the current sentence
        sentence_chunks = [
            chunk for chunk in global_output.alignments[0] 
            if (chunk.ref_start_idx <= start_idx <= chunk.ref_end_idx) or (chunk.ref_start_idx <= end_idx <= chunk.ref_end_idx)
            # if (start_idx <= chunk.ref_start_idx <= end_idx) or (start_idx <= chunk.ref_end_idx <= end_idx)
        ]
        print("sentence_chunk", sentence_chunks)
        # Get hypothesis words mapped to the refernece words for that sentence. Insertions are not included
        sentence_hyp_words = []
        for chunk in sentence_chunks:
            # if chunk.type != 'insert':
            chunk_idx_start = max(chunk.hyp_start_idx, adjusted_start_idx)
            chunk_idx_end = min(chunk.hyp_end_idx, adjusted_end_idx)
            sentence_hyp_words.extend(global_hyp_words[chunk_idx_start:chunk_idx_end])
        # hypothesis sentence mapped to the current reference sentence
        sentence_hyp_text = " ".join(sentence_hyp_words)
        
        
        sentence_output = process_words(sentence_ref_text, sentence_hyp_text)
        
        sentence_results.append({
            'reference': sentence_ref_text,
            'hypothesis': sentence_hyp_text,
            'wer': sentence_output.wer,
            'alignments': _serialize_alignment(sentence_output)
        })
    print("\n==== sentence_results ====")
    for sentence in sentence_results:
        print("REF: ", sentence["reference"])
        print("HYP: ", sentence["hypothesis"])
        # print(sentence["wer"])
        # print(sentence["alignments"], "\n")
    
    # Calculate phonemic analysis for sentences
    phonemic_analysis = calculate_phonemic_analysis(sentence_results, "sentences")
        
    # 4. Compile final result with overall metrics and per-sentence breakdown
    segment_wer = sum([sentence['wer'] for sentence in sentence_results]) / len(sentence_results)
    final_metrics = {
        #todo: calculate wer and cer with mean of each sentence and not globally
        'wer': min(segment_wer, 1.0),
        'visualize_alignment': visualize_alignment(global_output),
        'alignments_fda2': sentence_results,
        'phonemic_analysis': phonemic_analysis,
        # "global_alignments": _serialize_alignment(global_output)
    }
    
    return final_metrics


async def convert_audio_to_16khz(audio_path: Path, status_callback=None) -> Path:
    """Convert audio to 16kHz mono WAV format"""
    if status_callback:
        status_callback("converting", "Converting audio to 16kHz mono format with FFmpeg...")
    
    # Create a unique output filename to avoid overwriting input
    wav_file = audio_path.parent / f"{audio_path.stem}_converted.wav"
    
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-ac", "1", "-ar", "16000", str(wav_file)],
            check=True
        )
        
        # Simulate longer processing time
        await simulate_delay()
        
        if status_callback:
            status_callback("converted", "Audio conversion completed successfully")
            
        return wav_file
        
    except subprocess.CalledProcessError as e:
        if status_callback:
            status_callback("error", f"FFmpeg conversion failed: {str(e)}")
        raise

async def transcribe_with_whisper(wav_file: Path, status_callback=None) -> dict:
    """Transcribe audio using Whisper.cpp"""
    if status_callback:
        status_callback("transcribing", "Running Whisper transcription...")
    
    try:
        # Validate paths before running transcription
        validate_paths()
        
        whisper_out = await asyncio.to_thread(
            subprocess.check_output,
            [str(WHISPER_PATH), "-m", str(MODEL_PATH),
             "-f", str(wav_file),"-nt", "--output-json-full", "-of", str(wav_file), "-np"]
        )
        
        transcript = whisper_out.decode().strip()
       
        if status_callback:
            status_callback("transcribed", "Whisper transcription completed")
            
        return {"transcript": transcript}
        
    except FileNotFoundError as e:
        if status_callback:
            status_callback("error", f"Whisper setup error: {str(e)}")
        raise
    except subprocess.CalledProcessError as e:
        if status_callback:
            status_callback("error", f"Whisper transcription failed: {str(e)}")
        raise

async def analyze_formants_with_praat(wav_file: Path, status_callback=None) -> dict:
    """Analyze formants using Praat (placeholder implementation)"""
    if status_callback:
        status_callback("acoustic_analysis", "Running Praat formant analysis...")
    
    try:
        # TODO: Implement actual Praat formant analysis
        # For now, return mock data
        await simulate_delay()  # Simulate processing time
        
        formant_data = {
            "f1_mean": 500.0,
            "f2_mean": 1500.0,
            "f3_mean": 2500.0,
            "analysis_points": 100
        }
        
        # Simulate longer processing time
        await simulate_delay()
        
        if status_callback:
            status_callback("acoustic_analysis_completed", "Praat formant analysis completed")
            
        return {"formants": formant_data}
        
    except Exception as e:
        if status_callback:
            status_callback("error", f"Praat formant analysis failed: {str(e)}")
        raise

async def read_transcription_file(file_path: Path) -> str:
    """Read transcription from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading transcription file: {e}")
        return ""

# ---------------------------------------------------------------------------
# Intelligibility scoring functions
# ---------------------------------------------------------------------------

def calculate_intelligibility_scores(segment_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate intelligibility scores from segment results.
    
    Parameters
    ----------
    segment_results : Dict[str, Any]
        Dictionary containing results from all processed segments
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing intelligibility scores:
        - sentences_wer: Mean WER for sentence-based segments (weighted by number of sentences)
        - words_wer: Mean WER for word-based segments (weighted by number of words)  
        - total_wer: Mean of sentences and words scores
        
    Note: All WER values are capped at 1.0 (100% error rate)
    """
    total_wer = 0
    total_sentences_wer = 0
    total_words_wer = 0
    nb_sentences = 0
    nb_words = 0

    if len(segment_results) == 0:
        return {
            "total_wer": None, 
            "sentences_wer": None,
            "words_wer": None,
        }
    
    # Separate segments by intelligibility type
    for segment_id, segment_data in segment_results.items():
        if segment_data.get("segment_type") == "intelligibility":
            intelligibility_type = segment_data.get("intelligibility_type")
            alignments = segment_data.get("metrics", {}).get("alignments_fda2", [[]])
           
            if intelligibility_type == "sentences":
                for alignment in alignments:
                    wer = min(alignment['wer'], 1.0)  # Cap at 1.0
                    total_sentences_wer += wer
                    total_wer += wer
                    nb_sentences += 1
                nb_sentences += 1
            elif intelligibility_type == "words":
                for alignment in alignments[0]:
                    wer = min(alignment.get("wer", 0), 1.0)  # Cap at 1.0
                    total_words_wer += wer
                    total_wer += wer
                    nb_words += 1
            
    results = {
        "total_wer": min(total_wer / (nb_sentences + nb_words), 1.0) if (nb_sentences + nb_words) > 0 else None,
        "sentences_wer": min(total_sentences_wer / nb_sentences, 1.0) if nb_sentences > 0 else None,
        "words_wer": min(total_words_wer / nb_words, 1.0) if nb_words > 0 else None,
    }
    
    return results

async def run_full_pipeline(audio_path: Path, segments: Optional[List[AnalysisSegment]] = None, status_callback=None, job_id: str = None, metadata: Optional[Dict[str, Any]] = None) -> dict:
    """Run the complete audio analysis pipeline based on segment types"""
    try:
        if status_callback:
            status_callback("processing_started", "Starting audio analysis pipeline...")
        
        # Analyze segments to determine pipeline steps
        analysis_types = analyze_segment_types(segments or [])

        user_id = metadata.get("user_id")
        patient_id = metadata.get("patient_id")
        
        # Step 1: Always convert audio to 16kHz (required for both transcription and formant analysis)
        wav_file = await convert_audio_to_16khz(audio_path, status_callback)
        
        final_result = {}
        
        # Add metadata to final result
        final_result["metadata"] = {
            "job_id": job_id,
            "date": metadata.get("date") if metadata else None,
            "duration": metadata.get("duration") if metadata else 0,
            "user_id": user_id,
            "patient_id": patient_id,
            "analysis_types": analysis_types
        }

        # Step 2: Process each segment individually
        intelligibility_segment_results = {}
        acoustic_segment_results = {}
        
        if segments:
            for segment in segments:
                if status_callback:
                    status_callback("processing_segment", f"Processing segment: {segment.name}")
                
                segment_prefix = segment.id.split("-")[0]
                
                # Create trimmed audio file path
                if job_id and user_id:
                    segment_audio_dir = Path(f"data/uploads/{user_id}/{job_id}")
                    segment_audio_dir.mkdir(parents=True, exist_ok=True)
                    segment_audio_file = segment_audio_dir / f"{audio_path.stem}_{segment_prefix}.wav"
                else:
                    # Fallback for local tests without user/job id
                    segment_audio_file = wav_file.parent / f"{wav_file.stem}_{segment_prefix}.wav"
                
                # Trim audio for this segment
                trimmed_audio = await trim_audio(
                    wav_file, 
                    segment.timeRange.start, 
                    segment.timeRange.end,
                    segment_audio_file,
                    status_callback
                )
                
                segment_result = {
                    "segment_id": segment.id,
                    "segment_name": segment.name,
                    "segment_type": segment.type,
                    "time_range": {
                        "start": segment.timeRange.start,
                        "end": segment.timeRange.end
                    },
                }
                
                # Step 3: For intelligibility segments, run transcription and metrics
                if segment.type == "intelligibility":
                    # Transcribe the trimmed audio
                    transcription_result = await transcribe_with_whisper(trimmed_audio, status_callback)
                    transcription = transcription_result.get("transcript", "")

                    # Read transcription from file
                    #transcription_file = segment_audio_file.with_suffix('.txt')
                    #transcription = await read_transcription_file(transcription_file)
                    
                    intelligibility_type = "words" if len(segment.referenceData.words) > 0  else  "sentences" if len(segment.referenceData.sentences) > 0 else ""


                    # Get reference text 
                    if intelligibility_type == "sentences":
                        # Pass list of sentences for per-sentence analysis
                        reference_data = segment.referenceData.sentences
                        normalized_reference = [normalize_text(s) for s in reference_data]
                    else: # "words" or default
                        reference_data = " ".join(segment.referenceData.words)
                        normalized_reference = normalize_text(reference_data)
              
                    
                    # Normalize both transcription and reference
                    normalized_transcription = normalize_text(transcription)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(normalized_reference, normalized_transcription)
                    
                    # Add to segment result
                    segment_result.update({
                        "intelligibility_type": intelligibility_type,
                        "transcription": transcription,
                        "normalized_transcription": normalized_transcription,
                        "reference": reference_data,
                        "normalized_reference": normalized_reference,
                        "metrics": metrics,
                    })

                    # Store segment result
                    intelligibility_segment_results[segment.id] = segment_result
                
        
  
        # Step 5: Calculate intelligibility global scores
        if status_callback:
            status_callback("calculating_scores", "Calculating intelligibility scores...")
        
        intelligibility_scores = calculate_intelligibility_scores(intelligibility_segment_results)
        final_result["intelligibility_scores"] = intelligibility_scores
        
        # Step 6: Add segment results to final result
        final_result["intelligibility_results"] = intelligibility_segment_results if "intelligibility" in analysis_types else None
        
        # Step 7: Conditionally run formant analysis (commented out for now)
        # if pipeline_config["pipeline_steps"]["formant_analysis"]:
        #     formant_result = await analyze_formants_with_praat(wav_file, status_callback)
        #     final_result.update(formant_result)
        # else:
        #     if status_callback:
        #         status_callback("formant_analysis_skipped", "Formant analysis skipped - no acoustic/intelligibility segments found")
        
        # Clean up converted file
        try:
            wav_file.unlink()
        except:
            pass
        
        return final_result
        
    except Exception as e:
        # Clean up on error
        try:
            if 'wav_file' in locals():
                wav_file.unlink()
        except:
            pass
        raise

async def process_audio_orchestrator(job_id: str, audio_path: Path, processing_status: dict, segments: Optional[List[AnalysisSegment]] = None, metadata: Optional[Dict[str, Any]] = None):
    """Process audio with status updates using the full pipeline"""
    try:
        def update_status(status, message):
            processing_status[job_id].update({
                "status": status,
                "message": message
            })
        
        user_id = metadata.get("user_id")
        # Run the full pipeline with segments and job_id
        result = await run_full_pipeline(audio_path, segments, update_status, job_id, metadata)
        
        # Create results directory
        results_dir = Path("data") / "uploads" / user_id / job_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the result to a json file
        serializable_result = _make_json_serializable(result)
        with open(results_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(serializable_result, f, indent=4)
        
        # Update status: Completed
        processing_status[job_id].update({
            "status": "completed",
            "message": "All processing completed successfully",
            "result": serializable_result
        })
        
    except Exception as e:
        processing_status[job_id].update({
            "status": "error",
            "message": f"Error: {str(e)}"
        })

