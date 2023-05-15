# Date: 2023-5-13
# Author: Generated by GoCodeo.


import unittest
from unittest.mock import patch
from autogpt.commands.analyze_code import analyze_code

patch_func1 = 'autogpt.commands.analyze_code.call_ai_function'

class TestAnalyzeCode(unittest.TestCase):

    @patch(patch_func1)
    def test_positive_analyze_code(self, mock_call_ai_function):
        # Positive Test
        mock_call_ai_function.return_value = ["Suggestion 1", "Suggestion 2"]
        code = "def example_function():\n    pass"
        result = analyze_code(code)
        self.assertEqual(result, ["Suggestion 1", "Suggestion 2"])
        mock_call_ai_function.assert_called_once_with("def analyze_code(code: str) -> list[str]:", [code], "Analyzes the given code and returns a list of suggestions for improvements.")

    @patch(patch_func1)
    def test_negative_analyze_code(self, mock_call_ai_function):
        # Negative Test
        mock_call_ai_function.return_value = []
        code = "def example_function():\n    pass"
        result = analyze_code(code)
        self.assertEqual(result, [])
        mock_call_ai_function.assert_called_once_with("def analyze_code(code: str) -> list[str]:", [code], "Analyzes the given code and returns a list of suggestions for improvements.")

    @patch(patch_func1)
    def test_error_analyze_code(self, mock_call_ai_function):
        # Error Test
        mock_call_ai_function.side_effect = Exception("Error occurred")
        code = "def example_function():\n    pass"
        with self.assertRaises(Exception):
            analyze_code(code)
        mock_call_ai_function.assert_called_once_with("def analyze_code(code: str) -> list[str]:", [code], "Analyzes the given code and returns a list of suggestions for improvements.")

    @patch(patch_func1)
    def test_edge_analyze_code_empty_code(self, mock_call_ai_function):
        # Edge Test
        mock_call_ai_function.return_value = ["Suggestion 1", "Suggestion 2"]
        code = ""
        result = analyze_code(code)
        self.assertEqual(result, ["Suggestion 1", "Suggestion 2"])
        mock_call_ai_function.assert_called_once_with("def analyze_code(code: str) -> list[str]:", [code], "Analyzes the given code and returns a list of suggestions for improvements.")

if __name__ == "__main__":
    unittest.main()
