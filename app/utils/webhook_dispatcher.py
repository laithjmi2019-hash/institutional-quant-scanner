import requests
import streamlit as st

def get_n8n_url():
    try:
        return st.secrets["webhooks"]["n8n_url"]
    except:
        return None

def dispatch_trade_signal(ticker: str, action: str, quantity: int, details: dict):
    """
    Async n8n cloud-to-cloud execution pipeline.
    """
    webhook_url = get_n8n_url()
    if not webhook_url:
        print(f"n8n webhook not configured. Simulation only for {action} {quantity} {ticker}")
        return False
        
    payload = {
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "details": details
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        if response.status_code in [200, 201, 202]:
            return True
        else:
            print(f"Webhook failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"Webhook exception: {e}")
        return False
