import cv2
import random

# Generate consistent colors for each track ID
def get_color(track_id):
    random.seed(track_id)  # ensures same ID gets same color
    return [int(x) for x in random.choices(range(0, 256), k=3)]

def draw_tracks(frame, tracks):
    """
    Draws bounding boxes and track IDs on the frame.

    Args:
        frame: The current video frame (BGR image).
        tracks: List of DeepSORT track objects.
    """
    for track in tracks:
        try:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            track_id = track.track_id
            track_cls = getattr(track, 'class_name', 'person')

            color = get_color(track_id)

            # Draw rectangle around object
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw filled rectangle for text background
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 100, y1), color, -1)

            # Draw text (class + ID)
            cv2.putText(frame, f'{track_cls} ID:{track_id}', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            # Skip tracks with invalid data
            continue
