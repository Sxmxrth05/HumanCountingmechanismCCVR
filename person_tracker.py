from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Tuple, Any

class Tracker:
    """
    DeepSORT Tracker wrapper for tracking detected objects (e.g., people).
    """
    def __init__(self, max_age: int = 30, n_init: int = 3, max_iou_distance: float = 0.7):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance
        )

    def update(self, detections: List[Tuple[List[int], float, Any]], frame) -> List[Any]:
        """
        Update the tracker with new detections.

        Args:
            detections: List of tuples in the format ([x1, y1, x2, y2], confidence, class_id)
            frame: The current video frame (optional for appearance features)

        Returns:
            List of active track objects (confirmed tracks with valid IDs)
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []

        for track in tracks:
            if track.is_confirmed() and track.track_id is not None:
                # Optionally, attach class info from detection
                if hasattr(track, 'det_class') and track.det_class is None and len(detections) > 0:
                    # Assign class from first detection if available
                    track.det_class = detections[0][2]
                # Add a class_name attribute for drawing purposes
                if not hasattr(track, 'class_name'):
                    track.class_name = getattr(track, 'det_class', 'person')

                active_tracks.append(track)

        return active_tracks
