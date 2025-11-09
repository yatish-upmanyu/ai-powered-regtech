import unittest
from app.controller.parsers.parse_xsd import parse_xsd

class TestParseXSD(unittest.TestCase):
    
    def test_parse_xsd(self):
        elements = parse_xsd('data/old_version/EPC115-06_2021_V1.0_pacs.008.001.02_Update.xsd')
        self.assertIsInstance(elements, dict)
        self.assertGreater(len(elements), 0)

if __name__ == '__main__':
    unittest.main()
