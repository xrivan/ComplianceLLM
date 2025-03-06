import streamlit as st
import pandas as pd
from utils import write_message
from graph import initialize_workflow


SERVER_URL = st.secrets['SERVER_URL']

# Page Config
st.set_page_config("Company background searching", page_icon=":material/search:", layout="wide")

# Initialize necessary session state variables
session_vars = [
    "messages", "results", "company", "submitted",
    "analysis_process", "pending_companies", "processing", "llm"
]
for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = [] if var in ["messages", "results", "analysis_process", "pending_companies"] else False if var == "processing" else ""

with st.sidebar:
    st.session_state.llm = st.radio(
            "Select the LLM backend",
            ["Qwen-plus", "Llama3-70B","Llama3-8B","Deepseek","Qwen-turbo","Qwen-MAX","Llama3-70B-A","Deepseek-T"],
            captions=[
                "AliCloud",
                "Together.AI",
                "Together.AI",
                "Deepseek",
                "AliCloud",
                "AliCloud",
                "AliCloud",
                "Together.AI",
            ],
    )
# Initialize llm workflow
app = initialize_workflow()

# Submit button handling
def handle_submit(company_names):
    """Handle form submission and initialize processing."""
    companies = [name.strip() for name in company_names.split(";") if name.strip()]
    st.session_state.pending_companies = companies
    st.session_state.processing = True

# Upper Section: Input form and results table
upper_container = st.container()
with upper_container:
    with st.form(key='search_form'):
        company_input = st.text_input("Company Names (separated by ';')", key="company_input")
        submitted = st.form_submit_button("🚀 Search")
    
    if submitted and company_input:
        handle_submit(company_input)
    
    # Display results table
    if st.session_state.results:
        df = pd.DataFrame(st.session_state.results)
        st.dataframe(
                df,
                column_config={
                    "company_name": "Company",
                    "industry":"Industry",
                    "ref_url":st.column_config.LinkColumn("Reference URL"),
                    "screenshot_link":st.column_config.LinkColumn("Screenshot", display_text="Click to view"),
                    "html_link":st.column_config.LinkColumn("HTML Archive",display_text="Click to view"),
                    },
                hide_index=True,
        )

# Lower Section: Analysis process log
lower_container = st.container(height=600, border=True)
with lower_container:
    st.subheader("Analysis Process")
    for msg in st.session_state.analysis_process:
        st.write(msg)

# Process each company incrementally
if st.session_state.processing and st.session_state.pending_companies:
    company_name = st.session_state.pending_companies[0]
    final_html = final_screenshot = final_industry = final_ref_url = ""
    
    with st.spinner(f'Analyzing {company_name}...'):
        inputs = {"company_name": company_name}
        for output in app.stream(inputs, {"recursion_limit": 30}):
            for key, value in output.items():
                st.session_state.analysis_process.append(f"Node '{key}' for {company_name} started:")
                if value.get("interim_message"):
                    st.session_state.analysis_process.append(value["interim_message"])
                final_industry = value.get("industry", final_industry)
                final_ref_url = value.get("ref_url", final_ref_url)
                final_screenshot = value.get("screenshot_filename", final_screenshot)
                final_screenshot = final_screenshot.lstrip("./")
                final_html = value.get("html_filename", final_html)
                final_html = final_html.lstrip("./")
        
        # Save results and log final messages
        st.session_state.results.append({
            "company_name": company_name,
            "industry": final_industry,
            "ref_url": final_ref_url,
            'screenshot_link': f"{SERVER_URL}{final_screenshot}",
            'html_link': f"{SERVER_URL}{final_html}",
        })
        if final_industry:
            st.session_state.analysis_process.append(f'Industry: {final_industry}')
        if final_ref_url:
            st.session_state.analysis_process.append(f'Reference URL: {final_ref_url}')
    
    # Move to next company or stop processing
    st.session_state.pending_companies.pop(0)
    if not st.session_state.pending_companies:
        st.session_state.processing = False
    st.rerun()
