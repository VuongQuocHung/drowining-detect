import unittest
from app.alerts import AlertEngine
from app.settings import normalize_class

def detection(label="drowning", tid=1):
    return {"track_id": tid, "class": label, "confidence": .9}

class AlertsTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual([normalize_class(x) for x in ("Normal", " Swimming ", "Struggling", "Drowning", "person")],
                         ["swimming", "swimming", "struggling", "drowning", "unknown"])

    def test_low_fps_still_uses_seconds(self):
        engine = AlertEngine()
        engine.update([detection()], 0)
        self.assertTrue(engine.update([detection()], 1)[0]["alarm"])

    def test_drowning_threshold_and_repeat(self):
        engine = AlertEngine()
        pulses = []
        for i in range(49):
            person = engine.update([detection()], i*.25)[0]
            if person["alarm"]:
                pulses.append(i*.25)
        self.assertEqual(pulses, [1., 11.])

    def test_struggling_does_not_play_red_alarm(self):
        engine = AlertEngine()
        for i in range(9):
            person = engine.update([detection("struggling")], i*.25)[0]
        self.assertEqual(person["alert_level"], "struggling_alert")
        self.assertFalse(person["alarm"])

    def test_short_gap_pauses_evidence(self):
        engine = AlertEngine()
        engine.update([detection()], 0)
        engine.update([detection()], .25)
        self.assertEqual(engine.update([], .4)[0]["alert_level"], "lost")
        resumed = engine.update([detection()], .5)[0]
        self.assertEqual(resumed["observed_seconds"], .25)
        for timestamp in (.75, 1., 1.25):
            resumed = engine.update([detection()], timestamp)[0]
        self.assertTrue(resumed["alarm"])

    def test_long_gap_resets_pending_evidence(self):
        engine = AlertEngine()
        engine.update([detection()], 0)
        engine.update([detection()], .5)
        engine.update([], 1.1)
        person = engine.update([detection()], 1.2)[0]
        self.assertEqual(person["observed_seconds"], 0)
        self.assertEqual(person["alert_level"], "suspected")

    def test_independent_people_and_sessions(self):
        first, second = AlertEngine(), AlertEngine()
        first.update([detection()], 0)
        first.update([detection()], .5)
        result = first.update([detection(tid=2)], 1)
        self.assertEqual(next(p for p in result if p["track_id"] == 2)["alert_level"], "suspected")
        self.assertEqual(second.update([detection()], 1)[0]["observed_seconds"], 0)

    def test_recovery_requires_two_seconds_swimming(self):
        engine = AlertEngine()
        for i in range(5):
            engine.update([detection()], i*.25)
        for i in range(8):
            person = engine.update([detection("swimming")], 1.25+i*.25)[0]
            self.assertEqual(person["alert_level"], "drowning_alert")
        self.assertEqual(engine.update([detection("swimming")], 3.25)[0]["alert_level"], "normal")

    def test_long_missing_never_reports_safe_and_memory_is_bounded(self):
        engine = AlertEngine()
        engine.update([detection()], 0)
        self.assertEqual(engine.update([], 10)[0]["alert_level"], "lost")
        engine.update([], 31)
        self.assertFalse(engine.people)

if __name__ == "__main__":
    unittest.main()
