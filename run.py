from app.controller.parse_xsd import parse_xsd
from app.model.compare_elements import compare_elements
from ml.impact_analyzer import generate_impact_summary_and_test_scenarios
import pandas as pd

def main():
    # Parse XSD files
    old_elements = parse_xsd('app/resources/data/old_version/EPC115-06_2021_V1.0_pacs.008.001.02_Update.xsd')
    new_elements = parse_xsd('app/resources/data/new_version/EPC115-06_2023_V1.0_pacs.008.001.08_Update.xsd')
    
    # Compare elements
    report = compare_elements(old_elements, new_elements)
    
    # Convert report to DataFrame
    df_report = pd.DataFrame(report, columns=['Change Type', 'Element Path', 'Old Type', 'New Type', 'Annotation'])
    
    # Analyze impact and generate test scenarios
    impact_summary, test_scenarios = generate_impact_summary_and_test_scenarios(df_report)
    
    # Save the impact summary and test scenarios
    with open('app/resources/data/impacts/impact_summary.txt', 'w') as f:
        f.write(impact_summary)
        
    with open('app/resources/data/impacts/test_scenarios.txt', 'w') as f:
        f.write(test_scenarios)

if __name__ == "__main__":
    main()
