import cv2

class Counter:
    def __init__(self, line_position_x, line_color=(0, 0, 255), line_thickness=2):
        """
        Args:
            line_position_x: X-coordinate of the vertical dividing line
            line_color: BGR color of the line
            line_thickness: Thickness of the line
        """
        self.line_x = line_position_x
        self.line_color = line_color
        self.line_thickness = line_thickness

        # Current counts
        self.count_left = 0
        self.count_right = 0

    def update(self, tracks):
        """
        Update counts based on current positions of tracks.
        Args:
            tracks: list of active track objects
        """
        left = 0
        right = 0
        for track in tracks:
            x1, y1, x2, y2 = track.to_ltrb()
            center_x = int((x1 + x2) / 2)

            if center_x < self.line_x:
                left += 1
            else:
                right += 1

        self.count_left = left
        self.count_right = right

    def draw(self, frame):
        """
        Draw vertical line and current counts on the frame.
        """
        # Draw vertical line
        cv2.line(frame, (self.line_x, 0), (self.line_x, frame.shape[0]),
                 self.line_color, self.line_thickness)

        # Draw counts
        cv2.putText(frame, f'Left: {self.count_left}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Right: {self.count_right}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
