# app/utils.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from enum import Enum
import asyncio

class AnalysisType(str, Enum):
    """Analysis type enum - adjust values based on your frontend schema"""
    ACOUSTIC = "acoustic"
    INTELLIGIBILITY = "intelligibility"
    # Add more types as needed

class TimeRange(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., gt=0, description="End time in seconds")
    
    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("End time must be greater than start time")

class ReferenceData(BaseModel):
    """Reference data - flexible structure to accommodate different analysis types"""
    data: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow additional fields

class AnalysisSegment(BaseModel):
    id: str = Field(..., min_length=1, description="Unique segment identifier")
    name: str = Field(..., min_length=1, description="Analysis name is required")
    type: AnalysisType = Field(..., description="Type of analysis")
    timeRange: TimeRange = Field(..., description="Time range for the segment")
    # can also be undefined, so we need to make it optional
    referenceData: Optional[ReferenceData] = Field(None, description="Reference data for the analysis")
    createdAt: datetime = Field(..., description="Creation timestamp")
    updatedAt: datetime = Field(..., description="Last update timestamp")

class MockAnalysisSegment(AnalysisSegment):
    transcription: str = Field(..., description="Transcription of the segment")

def parse_segments(segments_json: str) -> Optional[List[AnalysisSegment]]:
    """
    Parse segments JSON string.
    
    Args:
        segments_json: JSON string containing analysis segments
        
    Returns:
        List of validated AnalysisSegment objects or None if parsing fails
    """
    try:
        # Parse JSON
        raw_segments = json.loads(segments_json)
        
        if not isinstance(raw_segments, list):
            print("Error: Segments must be an array")
            return None
        
        # Validate and parse each segment
        parsed_segments = []
        validation_errors = []
        
        for i, segment_data in enumerate(raw_segments):
            try:
                segment = AnalysisSegment(**segment_data)
                parsed_segments.append(segment)
            except Exception as e:
                validation_errors.append(f"Segment {i+1}: {str(e)}")
        
        # Print validation errors if any
        if validation_errors:
            print("=== VALIDATION ERRORS ===")
            for error in validation_errors:
                print(f"  {error}")
            print("=" * 25)
     
        
        return parsed_segments
        
    except json.JSONDecodeError as e:
        print(f"Error parsing segments JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing segments: {e}")
        return None

def print_raw_segments(segments_json: str) -> None:
    """
    Print raw segments without validation (fallback function).
    
    Args:
        segments_json: JSON string containing analysis segments
    """
    try:
        parsed_segments = json.loads(segments_json)
        print("=== RAW ANALYSIS SEGMENTS ===")
        print(f"Number of segments: {len(parsed_segments)}")
        
        for i, segment in enumerate(parsed_segments):
            print(f"\nSegment {i+1}:")
            print(f"  ID: {segment.get('id')}")
            print(f"  Name: {segment.get('name')}")
            print(f"  Type: {segment.get('type')}")
            print(f"  Time Range: {segment.get('timeRange')}")
            print(f"  Reference Data: {segment.get('referenceData')}")
            print(f"  Created At: {segment.get('createdAt')}")
            print(f"  Updated At: {segment.get('updatedAt')}")
        print("=" * 29)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing segments JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

async def simulate_delay(duration: float = 2.0) -> None:
    """
    Simulate processing delay for demonstration purposes.
    
    Args:
        step_name: Name of the processing step (for debugging)
        duration: Duration of the delay in seconds
    """
    await asyncio.sleep(duration)

def analyze_segment_types(segments: List[AnalysisSegment]) -> dict:
    """
    Analyze the types of segments to determine which pipeline steps to run.
    
    Args:
        segments: List of analysis segments
        
    Returns:
        List of analysis types
    """

    analysis_types = []

    if not segments:
        return analysis_types
    
    print(f"Segments: {segments}")
    
    segment_types = [segment.type for segment in segments]
    has_intelligibility = AnalysisType.INTELLIGIBILITY in segment_types
    has_acoustic = AnalysisType.ACOUSTIC in segment_types

    if has_intelligibility:
        analysis_types.append(AnalysisType.INTELLIGIBILITY)
    if has_acoustic:
        analysis_types.append(AnalysisType.ACOUSTIC)

    print(f"Analysis types: {analysis_types}")
    
    return analysis_types


def parse_mock_segments(segments_json: str) -> Optional[List[MockAnalysisSegment]]:
    """
    Parse segments JSON string.
    
    Args:
        segments_json: JSON string containing analysis segments
        
    Returns:
        List of validated AnalysisSegment objects or None if parsing fails
    """
    try:
        # Parse JSON
        raw_segments = json.loads(segments_json)
        
        if not isinstance(raw_segments, list):
            print("Error: Segments must be an array")
            return None
        
        # Validate and parse each segment
        parsed_segments = []
        validation_errors = []
        
        for i, segment_data in enumerate(raw_segments):
            try:
                segment = MockAnalysisSegment(**segment_data)
                parsed_segments.append(segment)
            except Exception as e:
                validation_errors.append(f"Segment {i+1}: {str(e)}")
        
        # Print validation errors if any
        if validation_errors:
            print("=== VALIDATION ERRORS ===")
            for error in validation_errors:
                print(f"  {error}")
            print("=" * 25)
     
        
        return parsed_segments
        
    except json.JSONDecodeError as e:
        print(f"Error parsing segments JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing segments: {e}")
        return None
