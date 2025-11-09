import unittest
import pandas as pd
from ml.analyzers.impact_analyzer import generate_impact_summary_and_test_scenarios

class TestImpactAnalyzer(unittest.TestCase):
    
    def test_generate_impact_summary_and_test_scenarios(self):
        data = {
            'Change Type': ['Added', 'Removed', 'Modified'],
            'Element Path': ['/Document/Elem1', '/Document/Elem2', '/Document/Elem3'],
            'Old Type': [None, 'Type2', 'Type3'],
            'New Type': ['Type1', None, 'Type4'],
            'Annotation': [None, None, None]
        }
        df_report = pd.DataFrame(data)
        
        impact_summary, test_scenarios = generate_impact_summary_and_test_scenarios(df_report)
        
        self.assertIsInstance(impact_summary, str)
        self.assertIsInstance(test_scenarios, str)

if __name__ == '__main__':
    unittest.main()
