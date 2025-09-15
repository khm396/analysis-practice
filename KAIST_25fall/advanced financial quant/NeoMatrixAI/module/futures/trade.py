import pandas as pd
import requests
from typing import Optional

def flash_close_position(USER_KEY: str, symbol: Optional[str], productType: str, holdSide: Optional[str]) -> pd.DataFrame:
    url = "https://bitgettrader.fin.cloud.ainode.ai/futures/trade/close-position"

    headers = {
        "API-KEY": USER_KEY,
        "Target-Email": None
    }

    json = {
        "symbol": symbol,
        "productType": productType,
        "holdSide": holdSide
    }

    response = requests.post(url=url, json=json, headers=headers)
    data = response.json()['data']

    rows = []
    # Success list processing
    for item in data.get('successList', []):
        orderId = item.get('orderId', '')
        clientOid = item.get('clientOid', '')
        result = "success"
        rows.append([orderId, clientOid, result, "", ""])

    # Failure list processing
    for item in data.get('failureList', []):
        orderId = item.get('orderId', '')
        clientOid = item.get('clientOid', '')
        result = "failure"
        errorMsg = item.get('errorMsg', '')
        errorCode = item.get('errorCode', '')
        rows.append([orderId, clientOid, result, errorMsg, errorCode])

    df = pd.DataFrame(rows, columns=["orderId", "clientOid", "result", "errorMsg", "errorCode"])
    return df
