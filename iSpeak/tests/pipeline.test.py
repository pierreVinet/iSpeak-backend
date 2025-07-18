

import asyncio
from pathlib import Path
import json
from iSpeak.pipeline import calculate_metrics, display_alignment, run_full_pipeline
from iSpeak.utils import parse_segments
from iSpeak.config import validate_paths

# Get project root directory (backend folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load segments as raw JSON string
with open(Path(__file__).parent / 'segments.json', 'r') as f:
    segments_raw = f.read()  # Read as string instead of parsing

async def test_run_full_pipeline():
    # First validate that whisper paths are correctly set up
    validate_paths()
    
    audio_path = PROJECT_ROOT / "data" / "samples" / "jfk.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Test audio file not found at {audio_path}")
    
    metadata = {"user_id":"5a6d4830-0662-4b8c-a659-a2b59fa3604d","date":"2025-07-11T08:50:44.245Z","duration":11}
    

    
    update_status = None
    job_id = "test"
    parsed_segments = parse_segments(segments_raw)  # Pass the raw JSON string
    print(parsed_segments)
    result = await run_full_pipeline(audio_path, parsed_segments, update_status, job_id, "user-test", metadata)
    
    assert result is not None
    int_results = result["intelligibility_results"]
    # for i, int_result in enumerate(int_results):
    #     if int_results[int_result]["intelligibility_type"] == "sentences":
    #         print("sentences")
    #         for alignment in int_results[int_result]["metrics"]["alignments_fda2"]:
    #             print(alignment["alignments"])
    #             print(display_alignment(alignment["alignments"]))
    #     else:
    #         print("words")
    #         print(int_results[int_result]["metrics"]["alignments_fda2"][0])
   


def test_calculate_metrics():
    truth = [
        "ask not",
        "your countri can do for",
        "ask what you can do for your country"
      ]
    hypothesis = "ask no what your country can do for you" 
    metrics = calculate_metrics(truth, hypothesis)
    # for sentence in truth:
        #print(sentence)
    # print(metrics['visualize_alignment'])
    print(metrics)


if __name__ == "__main__":
    asyncio.run(test_run_full_pipeline())
    # test_calculate_metrics()



