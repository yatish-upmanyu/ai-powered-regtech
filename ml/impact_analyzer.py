import pandas as pd
import google.generativeai as genai

# Read the API key and setup a Gemini client
with open("C:/Users/mohds/Desktop/Project-2024/Regulatory-Platform/app/resources/data/.gemini.txt") as f:
    key = f.read().strip()

genai.configure(api_key=key)

def generate_impact_summary_and_test_scenarios(df_report):
    added_elements = df_report[df_report['Change Type'] == 'Added']
    removed_elements = df_report[df_report['Change Type'] == 'Removed']
    modified_elements = df_report[df_report['Change Type'] == 'Modified']
    
    impact_prompt = (
        "You are a specialist in SEPA credit transfer messages. Based on the following changes in SEPA pacs.008 Credit Transfer messages from 2021 to 2023, generate a detailed impact summary. Explain how these changes affect the user and their applications.\n\n"
        f"Added elements: {added_elements.to_dict()}\n"
        f"Removed elements: {removed_elements.to_dict()}\n"
        f"Modified elements: {modified_elements.to_dict()}\n\n"
    )

    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
    impact_summary_response = model.generate_content(impact_prompt)
    
    test_scenarios_prompt = (
        "You are a testing expert. Generate BDD test scenarios based on the following changes in SEPA Credit Transfer Messages. Focus on clarity, specific element details, expected behavior, and additional notes for context.\n\n"
        f"Added elements: {added_elements.to_dict()}\n"
        f"Removed elements: {removed_elements.to_dict()}\n"
        f"Modified elements: {modified_elements.to_dict()}\n\n"
        )

    test_scenarios_response = model.generate_content(test_scenarios_prompt)

    # Extract text from response
    impact_summary = impact_summary_response.candidates[0].content.parts[0].text
    test_scenarios = test_scenarios_response.candidates[0].content.parts[0].text

    return impact_summary, test_scenarios
