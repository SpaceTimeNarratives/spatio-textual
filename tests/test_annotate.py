import unittest
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

from spatio_textual.annotate import annotate_text

class TestAnnotateText(unittest.TestCase):
    def test_basic_annotation(self):
        text = "The forest surrounds a quiet lake."
        result = annotate_text(text)
        print(f"{10*'-'}\n{result}\n{10*'-'}")
        self.assertIsInstance(result, dict)  # or list, depending on your implementation
        self.assertIn("entities", result)    # Adjust based on the structure of your output

if __name__ == '__main__':
    unittest.main()
