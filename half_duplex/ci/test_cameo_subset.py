import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cameo_subset import select, EMOTIONS
class CAMEOTests(unittest.TestCase):
    def test_french_and_quota(self):
        samples=[
          dict(language="French",dataset="CaFE",emotion="neutral",transcription="Bonjour"),
          dict(language="English",dataset="CaFE",emotion="anger",transcription="Hi"),
          dict(language="French",dataset="CaFE",emotion="neutral",transcription="Bonjour encore"),
          dict(language="French",dataset="CaFE",emotion="sadness",transcription="Au revoir")]
        self.assertEqual([x["emotion"] for x in select(samples,"cafe",1)],["neutral","sadness"])
    def test_coverage_contract(self):
        self.assertEqual(len(EMOTIONS),7)
if __name__=="__main__":unittest.main()
