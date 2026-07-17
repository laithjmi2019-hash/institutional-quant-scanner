import requests
import json
import base64
import os
import streamlit as st

GITHUB_REPO = "laithjmi2019-hash/institutional-quant-scanner"
FILE_PATH = "data/portfolio.json"

def get_github_token():
    try:
        return st.secrets["github"]["pat_token"]
    except:
        return None

def fetch_portfolio_state() -> dict:
    """
    Fetches the portfolio state from GitHub repository.
    """
    token = get_github_token()
    if not token:
        # Fallback to local if no token
        if os.path.exists(FILE_PATH):
            with open(FILE_PATH, 'r') as f:
                return json.load(f)
        return {"holdings": {}}
        
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {token}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content_b64 = response.json()['content']
        content_str = base64.b64decode(content_b64).decode('utf-8')
        return json.loads(content_str)
    else:
        # File doesn't exist yet or unauthorized
        return {"holdings": {}}

def sync_portfolio_state(state: dict):
    """
    Commits the updated portfolio state to GitHub.
    """
    token = get_github_token()
    if not token:
        # Fallback to local
        if not os.path.exists("data"):
            os.makedirs("data")
        with open(FILE_PATH, 'w') as f:
            json.dump(state, f, indent=4)
        return
        
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{FILE_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    
    # Need to get SHA of existing file to update it
    sha = None
    get_response = requests.get(url, headers=headers)
    if get_response.status_code == 200:
        sha = get_response.json()['sha']
        
    content_b64 = base64.b64encode(json.dumps(state, indent=4).encode('utf-8')).decode('utf-8')
    
    data = {
        "message": "Auto-sync portfolio state",
        "content": content_b64
    }
    if sha:
        data["sha"] = sha
        
    requests.put(url, headers=headers, json=data)
