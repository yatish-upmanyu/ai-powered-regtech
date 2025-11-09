import unittest
from src.comparers.compare_elements import compare_elements

class TestCompareElements(unittest.TestCase):
    
    def test_compare_elements(self):
        old_elements = {'/Document': ('DocumentType1', None)}
        new_elements = {'/Document': ('DocumentType2', None)}
        
        report = compare_elements(old_elements, new_elements)
        
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0][0], 'Modified')

if __name__ == '__main__':
    unittest.main()
