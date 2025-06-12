# utils/tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortWrapper:
    def __init__(self):
        self.tracker = DeepSort(max_age=30,
                                n_init=3,
                                max_cosine_distance=0.3,
                                nn_budget=None,
                                override_track_class=None)

    def update_tracks(self, detections, frame):
        """
        Inputs:
            detections: list of [x1, y1, x2, y2, conf, class]
            frame: current video frame (BGR)
        Returns:
            tracked_objects: list of objects with bbox, ID, etc.
        """
        # Convert YOLOv5 format to [xmin, ymin, width, height]
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            w, h = x2 - x1, y2 - y1
            formatted_detections.append(([x1, y1, w, h], conf, f"class_{int(cls)}"))

        # Run Deep SORT
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)

        return tracks
