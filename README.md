# ComplianceLLM

## Description
Web APP to investigate company background using LLM

## Features
- Accept multiple company names separated by ';'
- Output the industry of the company, and its website screenshot and source html 

## Installation
To install and set up the project, follow these steps:

```bash
git clone https://github.com/xrivan/ComplianceLLM.git
cd ComplianceLLM
pip install -r requirements.txt
```

##  Usage
- Create a secrets.toml in ./.streamlit/ directory
- Add the LLM API key in the secrets.toml, as well as the local server url http://<your_ip>:9875/
- Start the local http server to host the screenshot and html files: python3 http_server.py
- Start the streamlit application: streamlit run bot.py --server.port <your_port>
