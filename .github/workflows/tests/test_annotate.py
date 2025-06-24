import unittest
from spatio_textual.annotate import annotate_text

class TestAnnotateText(unittest.TestCase):
    def test_basic_annotation(self):
        text = "The forest surrounds a quiet lake."
        result = annotate_text(text)
        self.assertIsInstance(result, dict)  # or list, depending on your implementation
        self.assertIn("entities", result)    # Adjust based on the structure of your output

if __name__ == '__main__':
    unittest.main()
