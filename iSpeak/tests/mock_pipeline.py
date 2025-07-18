

import asyncio
import datetime
from pathlib import Path
import json


from typing import Any, Dict, List, Optional
import uuid
from iSpeak.pipeline import _make_json_serializable, calculate_intelligibility_scores, calculate_metrics, normalize_text
from iSpeak.utils import AnalysisSegment, parse_mock_segments, parse_segments

# Get project root directory (backend folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load segments as raw JSON string
with open(Path(__file__).parent / 'segments.json', 'r') as f:
    segments_raw = f.read()  # Read as string instead of parsing



async def run_mock_pipeline() -> dict:
    """Run the complete audio analysis pipeline based on segment types"""
 
    parsed_segments = parse_mock_segments(segments_raw)
        
    final_result = {}
    analysis_types = ["intelligibility"]

    user_id = "user-test"
    job_id = str(uuid.uuid4())
    
    # Add metadata to final result
    final_result["metadata"] = {
        "job_id": job_id,
        "date": datetime.datetime.now().isoformat(),
        "duration": 10,
        "user_id": user_id,
        "patient_id": "patient-test",
        "analysis_types": analysis_types
    }

    # Step 2: Process each segment individually
    intelligibility_segment_results = {}
    acoustic_segment_results = {}
    
    if parsed_segments:
        for segment in parsed_segments:
            segment_prefix = segment.id.split("-")[0]

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
                
                transcription = segment.transcription

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
        
        # Create results directory
        results_dir = Path("backend") / "data" / "uploads" / user_id / job_id
        print("saved to", results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the result to a json file
        serializable_result = _make_json_serializable(final_result)
        with open(results_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(serializable_result, f, indent=4)
        
        return final_result
        

if __name__ == "__main__":
    results = asyncio.run(run_mock_pipeline())
    # print(results)
