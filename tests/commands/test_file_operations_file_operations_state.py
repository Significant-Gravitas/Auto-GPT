# Date: 2023-5-13
# Author: Generated by GoCodeo.


import unittest
from unittest.mock import patch, mock_open
from typing import Dict
from io import StringIO
from autogpt.commands.file_operations import file_operations_state

class TestFileOperationsState(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="write|file1.txt|12345\nappend|file2.txt|67890\ndelete|file1.txt")
    def test_positive_case(self, mock_file):
        expected_output = {"file2.txt": "67890"}
        self.assertEqual(file_operations_state("log_path"), expected_output)

    @patch("builtins.open", new_callable=mock_open, read_data="write|file1.txt|12345\nappend|file2.txt|67890")
    def test_positive_no_delete(self, mock_file):
        expected_output = {"file1.txt": "12345", "file2.txt": "67890"}
        self.assertEqual(file_operations_state("log_path"), expected_output)

    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_positive_empty_log(self, mock_file):
        expected_output = {}
        self.assertEqual(file_operations_state("log_path"), expected_output)

    def test_error_file_not_found(self):
            expected_output = {}
            result = file_operations_state("non_existent_log_path")
            self.assertEqual(result, expected_output)

    @patch("builtins.open", new_callable=mock_open, read_data="write|file1.txt|12345\ninvalid_operation|file2.txt|67890")
    def test_error_invalid_operation(self, mock_file):
        with self.assertRaises(ValueError):
            file_operations_state("log_path")

    @patch("builtins.open", new_callable=mock_open, read_data="write|file1.txt|12345\nappend|file2.txt")
    def test_error_missing_checksum(self, mock_file):
        with self.assertRaises(ValueError):
            file_operations_state("log_path")

    @patch("builtins.open", new_callable=mock_open, read_data="write|file1.txt|12345\nappend")
    def test_error_missing_path_and_checksum(self, mock_file):
        with self.assertRaises(ValueError):
            file_operations_state("log_path")

if __name__ == "__main__":
    unittest.main()