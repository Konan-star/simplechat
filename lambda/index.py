# lambda/index.py
import json
import os
import re
import urllib3                    
from urllib3.util import Timeout
from urllib3.exceptions import HTTPError
from botocore.exceptions import ClientError

# -------------------------------------------------
# 追加：FastAPI 接続に使う情報
# -------------------------------------------------
FASTAPI_ENDPOINT = os.environ["FASTAPI"]             #こちらにcolab FASTAPIを入力する
FASTAPI_TIMEOUT  = float(os.environ.get("FASTAPI_TIMEOUT", "20"))
http = urllib3.PoolManager(
    headers={"Content-Type": "application/json"},
    timeout=Timeout(connect=5.0, read=FASTAPI_TIMEOUT)
)

# -------------------------------------------------
def extract_region_from_arn(arn):
    match = re.search('arn:aws:lambda:([^:]+):', arn)
    return match.group(1) if match else "us-east-1"

bedrock_client = None                   # 既存変数は残しておく
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

def lambda_handler(event, context):
    try:
        # ---------- 受信ログ ----------
        print("Received event:", json.dumps(event))
        
        # ---------- 認証情報 ----------
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # ---------- リクエスト解析 ----------
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])

        print("Processing message:", message)

        # ---------- 会話履歴 ----------
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": message})

        # ---------- 既存の Bedrock 形式をそのまま再利用 ----------
        bedrock_messages = []
        for msg in messages:
            if msg["role"] == "user":
                bedrock_messages.append({"role": "user", "content": [{"text": msg["content"]}]})
            else:
                bedrock_messages.append({"role": "assistant", "content": [{"text": msg["content"]}]})

        request_payload = {
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": 512,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        # ---------- FastAPI へ POST ----------
        print("POST", FASTAPI_ENDPOINT, "payload:", json.dumps(request_payload)[:500])
        api_resp = http.request(
            "POST",
            FASTAPI_ENDPOINT,
            body=json.dumps(request_payload).encode("utf-8"),
        )

        if api_resp.status != 200:
            raise RuntimeError(f"FastAPI returned {api_resp.status}: {api_resp.data[:300].decode()}")

        response_body = json.loads(api_resp.data.decode("utf-8"))
        print("FastAPI response:", json.dumps(response_body, default=str))

        # ---------- 応答取り出し ----------
        assistant_response = (
            response_body.get("response")        
            or response_body.get("assistant")      
            or response_body.get("answer")
        )
        if not assistant_response:
            raise ValueError("No response content from FastAPI")

        messages.append({"role": "assistant", "content": assistant_response})

        # ---------- 成功レスポンス ----------
        return {
            "statusCode": 200,
            "headers": _cors_headers(),
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }

    except (HTTPError, RuntimeError, ValueError) as error:
        print("Error:", str(error))
        return _error(str(error))
    except Exception as error:
        print("Unhandled Error:", str(error))
        return _error("Internal server error")


# -------------------------------------------------
# 共通ヘッダー／エラーレスポンス
# -------------------------------------------------
def _cors_headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "OPTIONS,POST"
    }

def _error(msg):
    return {
        "statusCode": 500,
        "headers": _cors_headers(),
        "body": json.dumps({"success": False, "error": msg})
    }
