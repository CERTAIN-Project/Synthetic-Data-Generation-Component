"""Simulation result export and formatting utilities."""

import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from .minio_manager import MinIOClient


class SimulationResultExporter:
    """Export simulation results to various formats."""
    
    @staticmethod
    def export_as_dataset(
        shared_history,
        topic_title: str,
        topic_summary: str,
        include_metadata: bool = True,
        metadata_records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Export simulation history as dataset JSON.
        
        Args:
            shared_history: SharedHistory object from simulation
            topic_title: Title of the discussion
            topic_summary: Summary of discussion
            include_metadata: Whether to include user metadata
            metadata_records: List of metadata dicts for actors
            
        Returns:
            Dataset formatted as dict
        """
        dataset_records = []
        
        # Extract posts (comments)
        posts = shared_history.posts
        metadata_map = {i: (metadata_records[i] if metadata_records else {}) for i in range(len(metadata_records or []))}
        
        for post in posts:
            actor_idx = post.get("actor_idx", 0)
            metadata = metadata_map.get(actor_idx) if include_metadata else None
            
            record = {
                "Id": post.get("id", ""),
                "Text": post.get("text", ""),
                "Metadata": metadata if metadata and include_metadata else {
                    "FirstName": None,
                    "LastName": None,
                    "Age": None,
                    "DateOfBirth": None,
                    "Nationality": None,
                    "Background": None
                }
            }
            dataset_records.append(record)
        
        return {
            "TopicTitle": topic_title,
            "TopicSummary": topic_summary,
            "Dataset": dataset_records,
            "ExportInfo": {
                "TotalComments": len(posts),
                "ExportedAt": datetime.now().isoformat(),
                "IncludesMetadata": include_metadata
            }
        }
    
    @staticmethod
    def to_json_string(result: Dict[str, Any], indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(result, indent=indent, ensure_ascii=False)


class AnonymousActorSimulationHandler:
    """Handle simulation with anonymous actors (no voting)."""
    
    @staticmethod
    def should_skip_voting(actor_has_metadata: bool) -> bool:
        """Determine if voting should be skipped."""
        return not actor_has_metadata
    
    @staticmethod
    def filter_events_for_anonymous(shared_history, anonymous_actor_indices: List[int]):
        """
        Filter events to remove voting from anonymous actors.
        
        Args:
            shared_history: SharedHistory object
            anonymous_actor_indices: Indices of anonymous actors
            
        Returns:
            Filtered events (only posts/replies, no votes)
        """
        filtered_events = []
        
        for event in shared_history.all_events():
            actor_idx = event.get("actor_idx")
            event_type = event.get("type")
            
            # Keep all posts/replies
            if event_type == "post":
                filtered_events.append(event)
            # Skip votes from anonymous actors
            elif event_type == "action" and actor_idx not in anonymous_actor_indices:
                filtered_events.append(event)
        
        return filtered_events


class MinIOResultUploader:
    """Upload simulation results to Minio."""
    
    def __init__(self, minio_client: MinIOClient):
        """Initialize with Minio client."""
        self.client = minio_client
        self.bucket_name = "simulation-results"
    
    def upload_result(
        self,
        result_json: str,
        filename: str,
        topic_name: str = ""
    ) -> bool:
        """
        Upload simulation result to Minio.
        
        Args:
            result_json: JSON string of result
            filename: Name for the file
            topic_name: Optional topic subfolder
            
        Returns:
            Success status
        """
        try:
            # Prepare temporary file
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                f.write(result_json)
                temp_path = f.name
            
            try:
                # Construct object name
                if topic_name:
                    object_name = f"simulation-results/{topic_name}/{filename}.json"
                else:
                    object_name = f"simulation-results/{filename}.json"
                
                # Upload
                success = self.client.upload_file(temp_path, self.bucket_name, object_name)
                
                if success:
                    print(f"[OK] Uploaded result to Minio: {object_name}")
                    return True
                else:
                    print(f"[ERROR] Failed to upload result to Minio")
                    return False
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        except Exception as e:
            print(f"[ERROR] Error uploading to Minio: {e}")
            return False


def create_mixed_dataset(
    posts_with_metadata: List[Dict[str, Any]],
    posts_anonymous: List[Dict[str, Any]],
    topic_title: str,
    topic_summary: str
) -> Dict[str, Any]:
    """
    Create dataset with mixed metadata and anonymous records.
    
    Args:
        posts_with_metadata: Posts from actors with metadata
        posts_anonymous: Posts from anonymous actors
        topic_title: Discussion topic title
        topic_summary: Discussion topic summary
        
    Returns:
        Combined dataset
    """
    all_posts = posts_with_metadata + posts_anonymous
    
    return {
        "TopicTitle": topic_title,
        "TopicSummary": topic_summary,
        "Dataset": all_posts,
        "Statistics": {
            "total_posts": len(all_posts),
            "with_metadata": len(posts_with_metadata),
            "anonymous": len(posts_anonymous),
            "anonymity_rate": f"{(len(posts_anonymous) / max(1, len(all_posts))) * 100:.1f}%"
        }
    }
