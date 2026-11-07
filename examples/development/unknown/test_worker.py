import unittest

class ConfigurationTests(unittest.TestCase):
    def test_read_settings(self):
        from worker import read_settings
        self.assertEqual(read_settings(), {"beta": "2"})
