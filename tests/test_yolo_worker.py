import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.yolo_worker import classify_detection


class YoloWorkerDetectionTests(unittest.TestCase):
    def test_person_is_classified_as_worker(self):
        result = classify_detection("person", 0.91)
        self.assertEqual(result["kind"], "worker")
        self.assertEqual(result["label"], "WORKER")

    def test_vehicle_is_classified_as_equipment(self):
        result = classify_detection("truck", 0.82)
        self.assertEqual(result["kind"], "equipment")
        self.assertEqual(result["label"], "TRUCK")

    def test_other_coco_objects_are_not_dropped(self):
        result = classify_detection("traffic light", 0.76)
        self.assertEqual(result["kind"], "equipment")
        self.assertEqual(result["label"], "TRAFFIC LIGHT")


if __name__ == "__main__":
    unittest.main()
