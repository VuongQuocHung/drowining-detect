import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from audit import audit
from evaluate_events import score
from train import choose

class DatasetTest(unittest.TestCase):
    def fixture(self, root):
        for index,split in enumerate(("train","train","val")):
            (root / "images" / split).mkdir(parents=True,exist_ok=True)
            (root / "labels" / split).mkdir(parents=True,exist_ok=True)
            Image.new("RGB",(64,64),(index*70,30,20)).save(root/"images"/split/f"{index}.jpg")
            (root/"labels"/split/f"{index}.txt").write_text(f"{index} .5 .5 .4 .4\n")
        (root/"labels/train/classes.txt").write_text("Swimming\nStruggling\nDrowning\n")

    def roboflow_fixture(self, root, names):
        # Roboflow YOLO export: {train,valid,test}/{images,labels} plus data.yaml.
        for index,split in enumerate(("train","train","valid","valid","test")):
            (root/split/"images").mkdir(parents=True,exist_ok=True)
            (root/split/"labels").mkdir(parents=True,exist_ok=True)
            Image.new("RGB",(64,64),(index*50,30,20)).save(root/split/"images"/f"{index}.jpg")
            (root/split/"labels"/f"{index}.txt").write_text(f"{index%2} .5 .5 .4 .4\n")
        (root/"data.yaml").write_text(f"nc: 2\nnames: {names}\n")

    def test_original_split_and_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.fixture(root)
            report=audit(root,output)
            self.assertEqual(report["total_images"],3)
            self.assertFalse(report["grouped"])
            self.assertEqual(report["split_counts"],{"train":2,"val":1})
            self.assertTrue(report["limitations"])
            self.assertTrue((output/"annotations.jpg").exists())

    def test_invalid_label_and_exact_leakage_block_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.fixture(root)
            (root/"labels/train/0.txt").write_text("7 .5 .5 .4 .4\n")
            with self.assertRaises(ValueError):
                audit(root,output)
            self.assertTrue(json.loads((output/"audit.json").read_text())["errors"])
            (root/"labels/train/0.txt").write_text("0 .5 .5 .4 .4\n")
            (root/"images/val/2.jpg").write_bytes((root/"images/train/0.jpg").read_bytes())
            with self.assertRaises(ValueError):
                audit(root,output)
            self.assertTrue(json.loads((output/"audit.json").read_text())["exact_cross_split"])
            Image.new("RGB",(64,64),(200,200,200)).save(root/"images/val/3.jpg")
            (root/"labels/val/3.txt").write_text("2 .5 .5 .4 .4\n")
            report=audit(root,output,drop_exact_duplicates=True)
            self.assertEqual(report["dropped_exact_duplicates"],["val/2.jpg"])
            self.assertEqual(report["split_counts"],{"train":2,"val":1})
            self.assertTrue((root/"images/val/2.jpg").exists())

    def test_grouped_split_never_crosses_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.fixture(root)
            groups=Path(folder)/"groups.csv"
            groups.write_text("image,group\ntrain/0.jpg,A\ntrain/1.jpg,B\nval/2.jpg,C\n")
            report=audit(root,output,groups)
            self.assertEqual(report["split_counts"],{"train":1,"val":1,"test":1})

    def test_roboflow_two_class_layout(self):
        import yaml
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.roboflow_fixture(root,"['Drowning', 'Normal']")
            report=audit(root,output)
            self.assertEqual(report["names"],["drowning","swimming"])
            self.assertEqual(report["split_counts"],{"train":2,"val":2,"test":1})
            self.assertEqual(report["split_class_boxes"]["val"],{"drowning":1,"swimming":1})
            data=yaml.safe_load((output/"data.yaml").read_text())
            self.assertEqual(data["names"],{0:"drowning",1:"swimming"})
            first=(output/"train.txt").read_text().splitlines()[0]
            # Ultralytics finds labels by swapping /images/ for /labels/ in each listed path.
            self.assertTrue(Path(first.replace("images","labels")).with_suffix(".txt").exists())

    def test_roboflow_water_rescue_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.roboflow_fixture(root,"['Drowning', 'float', 'swimmer']")
            self.assertEqual(audit(root,output)["names"],["drowning","float","swimming"])

    def test_roboflow_out_of_water_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.roboflow_fixture(root,"['Drowning', 'Person out of water', 'Swimming']")
            self.assertEqual(audit(root,output)["names"],["drowning","out_of_water","swimming"])

    def test_unknown_class_names_need_explicit_rename(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"dataset"; output=Path(folder)/"prepared"
            self.roboflow_fixture(root,"['drowning', 'person']")
            with self.assertRaisesRegex(ValueError,"--rename 'person=<label>'"):
                audit(root,output)
            self.assertEqual(audit(root,output,rename={"person":"swimming"})["names"],["drowning","swimming"])

    def test_selection_prefers_latency_only_within_one_point(self):
        def result(name, recall, precision, latency, map50=.9):
            return {"model":name,"map50":map50,"per_class":{"drowning":{"recall":recall,"precision":precision}},"latency_p50_ms":latency}
        slow=result("s",.95,.96,30)
        fast=result("n",.945,.955,10)
        self.assertEqual(choose([slow,fast])["model"],"n")
        fast["per_class"]["drowning"]["recall"] = .90
        self.assertEqual(choose([slow,fast])["model"],"s")
        # Underwater run 2026-10-26: n reached drowning recall 1.0 by over-predicting drowning.
        over=result("n",1.0,.685,20.2,map50=.755)
        balanced=result("s",.949,.765,20.6,map50=.816)
        self.assertEqual(choose([over,balanced])["model"],"s")

    def test_event_metrics_count_episodes_not_repeat_sound(self):
        truth={"video_id":"independent","duration_seconds":60,"events":[{"track_id":1,"start":2,"end":10}]}
        def event(t,level):
            return {"source_time":t,"detections":[{"track_id":1,"alert_level":level}]}
        result=score(truth,[event(3,"drowning_alert"),event(4,"drowning_alert"),event(11,"normal"),event(20,"drowning_alert")])
        self.assertEqual(result["matched"],1)
        self.assertEqual(result["delay_seconds"],[1])
        self.assertEqual(result["false_alarms_per_minute"],1)

if __name__ == "__main__":
    unittest.main()
