import pandas as pd
import requests

def all_positions(USER_KEY: str, productType: str, marginCoin: str) -> pd.DataFrame:
    url = "https://bitgettrader.fin.cloud.ainode.ai/futures/position/all-positions"
  
    headers = {
        "API-KEY": USER_KEY,
        "Target-Email": None
    }

    json = {
        "productType": productType,
        'marginCoin': marginCoin.upper()
    }

    response = requests.post(url, json=json, headers=headers)

    data = response.json()['data']
    df = pd.DataFrame(data)
    return df
