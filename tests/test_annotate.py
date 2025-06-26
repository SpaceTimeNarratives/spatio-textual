# import unittest
# import warnings
# warnings.simplefilter("ignore", category=FutureWarning)

# from spatio_textual.annotate import annotate_text

# class TestAnnotateText(unittest.TestCase):
#     def test_basic_annotation(self):
#         text = "Anne Frank was taken from Amsterdam to Auschwitz."
#         result = annotate_text(text)
#         print(f"{10*'-'}\n{result}\n{10*'-'}")

#         self.assertIsInstance(result, dict)  # or list, depending on your implementation
#         self.assertIn("entities", result)    # Adjust based on the structure of your output

#         expected_entities, verb_data = [
#             {'start_char': 0,  'token': "Anne Frank", 'tag': "PERSON"},
#             {'start_char': 26, 'token': "Amsterdam",  'tag': "CITY"},
#             {'start_char': 39, 'token': "Auschwitz",  'tag': "CAMP"},
#             ], [{'sent-id': 0, 'verb': 'taken', 'subject': 'Anne Frank', 'object': 'Amsterdam', 
#                  'sentence': 'Anne Frank was taken from Amsterdam to Auschwitz.'}]

#         self.assertEqual(result["entities"], expected_entities)
#         self.assertEqual(result["verb_data"], verb_data)

# if __name__ == '__main__':
#     unittest.main()

import unittest
import json
from pathlib import Path
import warnings
from spatio_textual.annotate import annotate_text

# Ignore a specific warning that may not be relevant to the test's outcome
warnings.simplefilter("ignore", category=FutureWarning)

class TestAnnotateText(unittest.TestCase):

    def setUp(self):
        """Load test cases from an external JSON file before running tests."""
        try:
            # Get the path to the JSON file in the same directory as the script
            data_file = Path(__file__).parent / "test_data.json"
            with open(data_file, 'r', encoding='utf-8') as f:
                self.test_cases = json.load(f)
        except FileNotFoundError:
            self.fail("The 'test_data.json' file was not found.")
        except json.JSONDecodeError:
            self.fail("Could not decode the 'test_data.json' file. Please check for syntax errors.")

    def test_text_annotation_from_file(self):
        """
        Tests the annotate_text function by loading examples from a JSON file.
        Each example is processed as a subtest for clear progress and reporting.
        """
        # Loop through each test case loaded from the file
        for i, case in enumerate(self.test_cases):
            # Use the text's first 50 chars in the subtest message for easy identification
            with self.subTest(i=i, text=case["text"][:50] + "..."):
                # Print progress for visual feedback during the run
                print(f"\n--- Running Test Case {i+1}: {case['text'][:50]}... ---")

                # Get the actual result from the function under test
                result = annotate_text(case["text"])

                # Print the result for manual inspection
                print(f"Result: {result}")

                # Assert that the structure is as expected
                self.assertIsInstance(result, dict)
                self.assertIn("entities", result)
                self.assertIn("verb_data", result)

                # Assert that the content matches the expected output
                self.assertEqual(result["entities"], case["expected_entities"])
                self.assertEqual(result["verb_data"], case["expected_verb_data"])

if __name__ == '__main__':
    # Running the test with verbosity level 2 for detailed progress
    unittest.main(verbosity=2)