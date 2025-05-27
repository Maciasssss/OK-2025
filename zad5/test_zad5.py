import unittest
import subprocess
import os
import sys
from pathlib import Path

try:
    # Assuming zad5.py is in the same directory
    from zad5 import SuffixTree, SuffixTreeNode
except ImportError:
    print("Error: Make sure zad5.py is in the same directory or in PYTHONPATH.")
    print("You might need to run: export PYTHONPATH=$PYTHONPATH:.")
    sys.exit(1)

ZAD5_SCRIPT_PATH = Path(__file__).resolve().parent / "zad5.py"
DATA_TXT_PATH = Path(__file__).resolve().parent / "data.txt"


class TestSuffixTreeLogic(unittest.TestCase):
    """
    Tests the core SuffixTree logic directly.
    """

    def _get_rotation(self, s_input):
        """Helper to get rotation using the SuffixTree class directly."""
        if not s_input:
            return ""
        if len(s_input) == 1:
            return s_input

        terminal_char = chr(1)
        if terminal_char in s_input:
            raise ValueError(
                "Input for direct SuffixTree test should not contain terminal char.")

        text_for_tree = s_input + s_input + terminal_char
        st = SuffixTree(text_for_tree)
        return st.find_lexicographically_smallest_rotation(len(s_input))

    def test_empty_string_logic(self):
        self.assertEqual(self._get_rotation(""), "")

    def test_single_char_string_logic(self):
        self.assertEqual(self._get_rotation("a"), "a")
        self.assertEqual(self._get_rotation("z"), "z")

    def test_banana(self):
        self.assertEqual(self._get_rotation("banana"), "abanan")

    def test_abracadabra(self):
        self.assertEqual(self._get_rotation("abracadabra"), "aabracadabr")

    def test_aaaaa(self):
        self.assertEqual(self._get_rotation("aaaaa"), "aaaaa")

    def test_abcde(self):
        self.assertEqual(self._get_rotation("abcde"), "abcde")

    def test_cba(self):
        self.assertEqual(self._get_rotation("cba"), "acb")

    def test_mississippi(self):
        self.assertEqual(self._get_rotation("mississippi"),
                         "imississipp")

    def test_topcoder(self):
        self.assertEqual(self._get_rotation(
            "topcoder"), "codertop")

    def test_unique_chars_reverse_order(self):
        self.assertEqual(self._get_rotation("zyxw"), "wzyx")

    def test_repeating_pattern(self):
        self.assertEqual(self._get_rotation("ababa"), "aabab")

    def test_long_simple_repeat(self):
        self.assertEqual(self._get_rotation("zzza"), "azzz")


class TestZad5ScriptExecution(unittest.TestCase):
    """
    Tests the zad5.py script by running it as a subprocess.
    This checks file I/O, main logic flow, and error handling.
    """

    def tearDown(self):
        """Clean up data.txt after each test if it was created."""
        if DATA_TXT_PATH.exists():
            os.remove(DATA_TXT_PATH)

    def _run_script_with_input(self, input_string):
        """Helper to write to data.txt and run zad5.py."""
        with open(DATA_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(input_string)

        process = subprocess.run(
            [sys.executable, str(ZAD5_SCRIPT_PATH)],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        return process

    def test_script_empty_input(self):
        process = self._run_script_with_input("")
        self.assertEqual(process.stdout.strip(), "")
        self.assertEqual(process.returncode, 0)
        self.assertIn("Computation:", process.stderr)

    def test_script_single_char_input(self):
        process = self._run_script_with_input("a")
        self.assertEqual(process.stdout.strip(), "a")
        self.assertEqual(process.returncode, 0)

    def test_script_banana_input(self):
        process = self._run_script_with_input("banana")
        self.assertEqual(process.stdout.strip(), "abanan")
        self.assertEqual(process.returncode, 0)

    def test_script_abracadabra_input(self):
        process = self._run_script_with_input("abracadabra")
        self.assertEqual(process.stdout.strip(), "aabracadabr")
        self.assertEqual(process.returncode, 0)

    def test_script_cba_input(self):
        process = self._run_script_with_input("cba")
        self.assertEqual(process.stdout.strip(), "acb")
        self.assertEqual(process.returncode, 0)

    def test_script_mississippi_input(self):
        process = self._run_script_with_input("mississippi")
        self.assertEqual(process.stdout.strip(), "imississipp")
        self.assertEqual(process.returncode, 0)

    def test_script_input_with_terminal_char_error(self):
        terminal_char = chr(1)
        input_str = f"abc{terminal_char}def"
        process = self._run_script_with_input(input_str)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Error: Terminal character", process.stderr)
        self.assertEqual(process.stdout.strip(), "")

    def test_script_input_too_long_error(self):
        max_len_from_script = 5 * 10**5
        # Create a string that is definitely too long.
        # Writing such a long string to disk for a test can be slow.
        try:
            long_string = "a" * (max_len_from_script + 1)
            process = self._run_script_with_input(long_string)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Error: Input string length", process.stderr)
            self.assertIn("exceeds the maximum allowed length", process.stderr)
            self.assertEqual(process.stdout.strip(), "")
        except MemoryError:
            self.skipTest(
                f"Skipping too_long test due to MemoryError creating string of length {max_len_from_script + 1}")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
