import streamlit as st
import json
import os
from langchain_openai import ChatOpenAI
from googlesearch import search
from langchain_core.tools import tool, Tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Set
from playwright.sync_api import sync_playwright
from datetime import datetime
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

DEEPSEEK_API_KEY = st.secrets['DEEPSEEK_API_KEY']
DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
LLAMA_API_KEY = st.secrets['TOGETHER_API_KEY']
LLAMA_API_URL = "https://api.together.xyz/v1"
LLAMA_70B_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
LLAMA_8B_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEEPSEEK_TOA_MODEL = "deepseek-ai/DeepSeek-V3"
DOUBAO_API_KEY = st.secrets['ARK_API_KEY']
DOUBAO_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_LITE_MODEL = "ep-20241218141514-c6rqc"
DOUBAO_PRO_MODEL = "ep-20241218120035-5226n"
DASHSCOPE_API_KEY = st.secrets['DASHSCOPE_API_KEY']
DASHSCOPE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_ALI_MODEL = "deepseek-v3"
LLAMA_70B_ALI_MODEL = "llama3.3-70b-instruct"
LLAMA_8B_ALI_MODEL = "llama3-8b-instruct"
QWEN_PLUS_MODEL = "qwen-plus"
QWEN_TURBO_MODEL = "qwen-turbo"
QWEN_MAX_MODEL = "qwen-max"


# Ensure st.session_state.llm is initialized
if 'llm' not in st.session_state:
    st.session_state.llm = 'Qwen-plus'  # Default value

# Function to initialize the LLM
def initialize_llm():
    llm_value = st.session_state.llm

    # Mapping of llm values to corresponding models, api keys, and base urls
    llm_mapping = {
        "Llama3-70B": (LLAMA_70B_MODEL, LLAMA_API_KEY, LLAMA_API_URL),
        "Llama3-8B": (LLAMA_8B_MODEL, LLAMA_API_KEY, LLAMA_API_URL),
        "Deepseek-T": (DEEPSEEK_TOA_MODEL, LLAMA_API_KEY, LLAMA_API_URL),
        "Deepseek": (DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_API_URL),
        "Doubao-pro": (DOUBAO_PRO_MODEL, DOUBAO_API_KEY, DOUBAO_API_URL),
        "Doubao-lite": (DOUBAO_LITE_MODEL, DOUBAO_API_KEY, DOUBAO_API_URL),
        "Deepseek-A": (DEEPSEEK_ALI_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_API_URL),
        "Llama3-70B-A": (LLAMA_70B_ALI_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_API_URL),
        "Llama3-8B-A": (LLAMA_8B_ALI_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_API_URL),
        "Qwen-plus": (QWEN_PLUS_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_API_URL),
        "Qwen-turbo": (QWEN_TURBO_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_API_URL),
        "Qwen-MAX": (QWEN_MAX_MODEL, DASHSCOPE_API_KEY, DASHSCOPE_API_URL),
    }

    model, api_key, base_url = llm_mapping.get(llm_value)

    return ChatOpenAI(
        temperature=0,
        max_tokens=4096,
        timeout=None,
        max_retries=2,
        model=model,
        api_key=api_key,
        base_url=base_url
    )



### Search tool
@tool
def search_google(query: str) -> str:
    """Perform a Google search for the given query and return the top results with titles, URLs, and descriptions.
    Use this to find relevant information about a company or topic."""
    results = []
    for result in search(query, num_results=10, advanced=True):
        results.append(f"Title: {result.title}\nURL: {result.url}\nDescription: {result.description}\n")
    return "\n\n".join(results)

### Crawler tool
@tool
def fetch_page_content(url: str) -> str:
    """
    Fetch the entire content of a webpage given its URL using Playwright's Sync API.
    This function includes measures to handle Cloudflare's anti-bot protection.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The entire content of the webpage as a string.
    """
    with sync_playwright() as p:
        # Launch a headless browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Set a realistic User-Agent header using extra_http_headers
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        page.set_extra_http_headers({"User-Agent": user_agent})

        # Navigate to the URL and wait for the page to load
        page.goto(url, wait_until="networkidle")

        # Check if Cloudflare's "Checking your browser" page is displayed
        if "Checking your browser" in page.title():
            print("Cloudflare challenge detected. Waiting for it to resolve...")
            # Wait for the challenge to complete (adjust timeout as needed)
            page.wait_for_selector("body", state="attached", timeout=30000)

        # Scroll to the bottom of the page to get total height
        total_height = page.evaluate("document.body.scrollHeight")

        # Set the viewport size to cover the entire page height
        page.set_viewport_size({"width": 1920, "height": total_height})

        # Get the entire content of the page
        content = page.content()

        # Set the directory to save the screenshot
        screenshot_dir = "./png"
        html_dir = "./html"
        os.makedirs(screenshot_dir, exist_ok=True)  # Create the directory if it doesn't exist
        os.makedirs(html_dir, exist_ok=True)

        # File name handling
        domain_name = url.replace("http://", "").replace("https://", "").replace("/","_")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Save html with the specified filename format
        html_filename = f"{html_dir}/{domain_name}_{timestamp}.html"
        with open(html_filename, "w", encoding="utf-8") as html_file:
            html_file.write(content)

        # Capture screenshot and save with the specified filename format
        screenshot_filename = f"{screenshot_dir}/{domain_name}_{timestamp}.png"
        page.screenshot(path=screenshot_filename)


        # Close the browser
        browser.close()

        return content,screenshot_filename, html_filename

### LLM agents,validate_website
# Define the data model for the official website results
class OfficialWebsite(BaseModel):
    """Model for representing a possible official website."""
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    description: str = Field(description="Description of the search result")

class OfficialWebsites(BaseModel):
    """Model for representing a list of possible official websites."""
    websites: List[OfficialWebsite] = Field(description="List of possible official websites")

# validate_website initialization function
def initialize_website_evaluator():
    
    # Initialize llm
    llm = initialize_llm()
    # Configure the LLM to output structured data
    structured_llm = llm.with_structured_output(OfficialWebsites)

    # Prompt for evaluating official websites
    system = """You are an expert at identifying official websites of companies. \n
        Given a search result, evaluate whether the URL and description are likely to be the company's official website. \n
        Return a list of possible official websites in JSON format with the following keys: title, url, and description."""
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Company name: {company_name} \n\n Search results: \n\n {search_results}"),
        ]
    )

    # Chain to evaluate search results
    website_evaluator = evaluation_prompt | structured_llm
    return website_evaluator

### LLM agent, investigator
# Define the data model for industry investigation results
class CompanyIndustry(BaseModel):
    """Model for representing company industry investigation results"""
    industry: str = Field(description="Identified industry sector and specialization. Use format 'Industry - Specialization' or 'Unknown'")
    ref_url: str = Field(description="Reference URL where information was found. 'NA' if not applicable")
    next_url: List[str] = Field(description="List of next URLs to investigate if current page inconclusive. Empty list if not applicable")

# investigator initialization function
def initialize_investigator():
    # Initialize llm
    llm = initialize_llm()
    investigate_structured_llm = llm.with_structured_output(CompanyIndustry)

    # System prompt for commercial investigation
    system = """You are a commercial investigation expert analyzing company websites. 
    Let's think step by step.
    Your tasks:
    1. Look in html <body> block and analyze the content to determine:
    - Industry sector (e.g., Manufacturing, Technology, Finance)
    - Specific business activities (e.g., Robotics, AI Software, Investment Banking), where it's possible, quote the source description here.
    - Possible links on the website that may contain related information (e.g., /about, About Us, Products) 
    - ALWAYS check the about page if you don't find the information in home page

    2. Follow these decision rules:
    a) If conclusive information exists:
        - Set industry to "Industry - Specialization"
        - Set ref_url to current URL
        - Set next_url to an empty list ([])

    b) If inconclusive but find promising links in the same website domain (e.g., /about, About Us, Products):
        - Set industry and ref_url to "NA"
        - Set next_url to a list of most relevant internal links
        - Make sure the internal links are FULL website links(start with http: or https:) but not relative links like /about.html.

    c) If completely inconclusive with no promising next links:
        - Set industry and ref_url to "NA"
        - Set next_url to an empty list ([])

    3. List of Industries for reference:
    Agriculture
    Automotive
    Banking
    Biotechnology
    Construction
    Education
    Energy
    Entertainment
    Fashion
    Food and Beverage
    Healthcare
    Information Technology
    Insurance
    Manufacturing
    Media
    Real Estate
    Retail
    Telecommunications
    Transportation
    Travel and Tourism
    """


    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Website Content:\n{raw_content}\n\nCurrent URL: {url}\nAnalysis:")
    ])

    investigation_chain = prompt_template | investigate_structured_llm
    return investigation_chain

### Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        company_name: str, the name of the company input by the user for searching.
        search_result: str, the search result returned by the web search.
        official_website: List[dict], a list of dictionaries representing official websites.
        next_url: List[str], a stack of URLs to be processed (last-added, first-removed).
        raw_content: str, web page content returned from web crawler.
        industry: str, the industry of the company, based on the web page content
        ref_url: str, the URL which contains the industry information
        interim_message: str, field for procedural messages
        screenshot_filename: str, path and filename for locally saved screenshot
        html_filename: str, path and filename for locally saved html
        visited_urls: Set[str], a set of URLs that have been visited or queued for processing.
    """
    company_name: str
    search_result: str
    official_website: List[dict]
    next_url: List[str]
    raw_content: str
    industry: str
    ref_url: str
    interim_message: str
    screenshot_filename: str
    html_filename: str
    visited_urls: Set[str]

def web_search(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, search_result, that contains qeury results from web search
    """
    # Initialize visited_url as empty set
    visited_urls=set()

    query = state["company_name"]
    interim_message =f"---WEB SEARCH for {query} STARTED: ---  \n"

    # Google Search
    search_results = search_google(query)
    interim_message +="---WEB SEARCH DONE---"
    return {"search_result": search_results, "company_name": query, "interim_message": interim_message, "visited_urls":visited_urls,}


def validate_website(state):
    """
    Determines a list of possible official websites from web search results.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, official_website, that contains list of possible official websites. And next_url, a list of url to investigate.
    """
    print("---CHECK WEB SEARCH RESULT---")
    interim_message ="---CHECK WEB SEARCH RESULT---  \n"
    search_results = state["search_result"]
    company_name = state["company_name"]

    # Evaluate the search results using the LLM
    website_evaluator = initialize_website_evaluator()
    evaluation_result = website_evaluator.invoke({
        "search_results": search_results,
        "company_name": company_name,
    })
    # Parse the JSON output into a Python dictionary
    data = evaluation_result.dict()

    # To store url for further process
    url_to_process = []

    # Print the formatted output
    print("Finding possible company website(s):")
    interim_message += "Finding possible company website(s):  \n"
    for website in data["websites"]:
        print(f"\n{website['title']}\n{website['url']}\n{website['description']}")
        interim_message += f"\n{website['title']}\n{website['url']}\n{website['description']}  \n"
        url_to_process.append(website['url'])
    print("---CHECK WEB SEARCH RESULT DONE---  \n")
    interim_message += "---CHECK WEB SEARCH RESULT DONE---"

    #Reverse so that first one is processed first.
    url_to_process = list(reversed(url_to_process))
    return {"official_website": data["websites"], "company_name": company_name, "search_result": search_results, "next_url":url_to_process, "interim_message": interim_message,}


def crawler(state):
    """
    Get a url from next_url, and return the web page content.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, or update the key, raw_content, from crawler result
    """
    url_to_process = state["next_url"]
    url = url_to_process[-1]
    print(f"---CRAWLING WEBSITE {url}---")
    interim_message = f"---CRAWLING WEBSITE {url}---  \n"
    page_content, screenshot_filename, html_filename = fetch_page_content(url)
    html_filename
    if len(page_content) >= 4096:
        page_content = page_content[:4096]
    interim_message += "---CRAWLING WEBSITE DONE---  \n"
    return {
            "raw_content": page_content, 
            "screenshot_filename": screenshot_filename, 
            "html_filename": html_filename,
            "next_url": url_to_process, 
            "interim_message": interim_message,}
    print(f"---CRAWLING WEBSITE DONE---")

def investigator(state):
    """
    Investigate the web page content, and determine whether the industry of the company can be decided.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Ney key added to state, industry, the industry of the company; ref_url, the url with the industry information; and update next_url if more urls to investigate.
    """

    print("---INVESTIGATING WEBSITE CONTENT---")
    interim_message = "---INVESTIGATING WEBSITE CONTENT---  \n"
    url_to_process = state["next_url"]
    # Remove url from process list
    url = url_to_process.pop()
    # Get visited list from State
    visited_urls = state["visited_urls"]
    # Add url to visited list
    visited_urls.add(url)
    # Get content and investigate
    raw_content = state["raw_content"]
    investigation_chain = initialize_investigator()
    result = investigation_chain.invoke({
        "url": url,
        "raw_content": raw_content
    })

    # Parse the JSON output into a Python dictionary
    data = result.dict()

    # Extend url_to_process with data["next_url"] in reverse order
    for next_url in reversed(data["next_url"]):
        if next_url not in visited_urls:
            url_to_process.append(next_url)
    print("---INVESTIGATING WEBSITE CONTENT DONE---")
    interim_message += "---INVESTIGATING WEBSITE CONTENT DONE---  \n"
    return {"industry": data["industry"], "ref_url": data["ref_url"], "next_url": url_to_process, "visited_urls":visited_urls, "interim_message": interim_message,}


### Edges


def router(state):
    """
    Determines whether the industry is found, or further investigation is needed.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS INVESTIGATION RESULT---")
    industry = state["industry"]

    if industry == "NA" and state["next_url"]:
        print(
            "---DECISION: INDUSTRY INFO NOT FOUND, CONTINUE TO INVESTIGATE---"
        )
        return "crawler"
    else:
        # The industry is either decided or unknown.
        print("---DECISION: FINISH---")
        return END

def initialize_workflow():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("validate_website", validate_website)  # decide official websites
    workflow.add_node("crawler", crawler)  # crawl web content
    workflow.add_node("investigator", investigator)  # investigate company industry info from web content

    # Build graph
    workflow.add_edge(START, "web_search")
    workflow.add_edge("web_search", "validate_website")

    workflow.add_edge("validate_website","crawler")
    workflow.add_edge("crawler","investigator")
    workflow.add_conditional_edges(
        "investigator",
        router,
        {
            "crawler": "crawler",
            END: END,
        },
    )

    # Compile
    return workflow.compile()

