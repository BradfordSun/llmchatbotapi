import time
import datetime
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
import json
import boto3
from botocore.exceptions import ClientError
import logging
from fastapi.middleware.cors import CORSMiddleware
from boto3.dynamodb.conditions import Key, Attr
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the environment variables from `.env` file.

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# 当前时间
now = datetime.datetime.now()
# 30天前
thirty_days_ago = now - datetime.timedelta(days=30)
# 转换为毫秒级的Unix时间戳
current_timestamp = int(now.timestamp() * 1000)
thirty_days_ago_timestamp = int(thirty_days_ago.timestamp() * 1000)

# 跨域相关设置
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 创建 AWS Bedrock boto3 客户端
def create_bedrock_client():
    return boto3.client('bedrock-runtime', aws_access_key_id=os.getenv("bedrock_ak"),
                        aws_secret_access_key=os.getenv("bedrock_sk"), region_name="us-east-1")


@app.post("/stream-response/")
async def stream_response(request: Request):
    data = await request.json()
    api_key = data.get("api_key")
    if api_key != os.getenv("api_key"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # image_path = data.get("image_path")

    is_new_session = data.get("is_new_session")
    session_id = data.get("session_id")
    session_name = data.get("session_name")
    start_timestamp = data.get("start_timestamp")
    user = data.get("user")
    model = data.get("model")
    system_prompt = data.get("system_prompt")
    input_text = data.get("input_text")
    max_tokens = int(100000)  # 默认值为100000，可以根据需要调整
    temperature = data.get("temperature_value")
    bedrock_client = create_bedrock_client()
    return StreamingResponse(
        stream_model_response(bedrock_client, is_new_session, session_id, session_name, start_timestamp, user, model,
                              system_prompt, input_text, max_tokens, temperature),
        media_type="text/event-stream")


async def stream_model_response(bedrock_client, is_new_session, session_id, session_name, start_timestamp, user, model,
                                system_prompt, input_text, max_tokens, temperature):
    try:
        dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.getenv("ddb_ak"),
                                  aws_secret_access_key=os.getenv("ddb_sk"),
                                  region_name='us-east-1')
        session_table = dynamodb.Table(os.getenv("session_table"))
        chat_history_table = dynamodb.Table(os.getenv("chat_history_table"))
        # 记录一个收到前端消息的时间
        received_message_timestamp = int(round(time.time() * 1000))
        messages = []
        if is_new_session:
            # 新的session的话就要在ddb的session表里面创建记录，最后更新时间先取收到消息的时间（因为必须指定个值）
            session_item = {
                'sessionID': session_id,
                'sessionName': session_name,
                'startTimestamp': start_timestamp,
                'lastUpdateTimestamp': received_message_timestamp,
                'user': user,
                'model': model,
            }
            session_table.put_item(Item=session_item)
        else:
            # 不是新chat的话就去读取一下chat history表的历史记录，放到内存里。这里不要显式指定session_id的类型为S，否则会报错
            response = chat_history_table.query(
                KeyConditionExpression='sessionID = :sid',
                ExpressionAttributeValues={':sid': session_id},
                ScanIndexForward=True  # True 为正序排列
            )
            for item in response['Items']:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": item.get('userMessage', '')}]
                })
                messages.append({
                    "role": "assistant",
                    "content": item.get('assistantMessage', '')
                })
        # 最后要追加一个最新的提问，加上之前读取的所有历史记录（如果是历史会话），新会话的话就只包含这个用户提问。
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": input_text}]
        })
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": messages
        })

        response = bedrock_client.invoke_model_with_response_stream(body=body,
                                                                    modelId="anthropic.claude-3-sonnet-20240229-v1:0")
        # 记录AI返回的消息（将流式传输所有块拼到一起）
        assistant_message = ""
        for event in response.get("body"):
            chunk = json.loads(event["chunk"]["bytes"])
            # 这段可以输出token？
            # if chunk['type'] == 'message_delta':
            #     yield f"Output tokens: {chunk['usage']['output_tokens']}\n"
            if chunk['type'] == 'content_block_delta' and chunk['delta']['type'] == 'text_delta':
                # 文本的话就流式传输yield到前端，并且将返回的块追加到消息记录
                assistant_message += chunk['delta']['text']
                yield chunk['delta']['text']
            # 以下表示写完了，就更新下session表里的lastUpdateTimestamp
            elif chunk['type'] == 'message_delta':
                sent_message_timestamp = int(round(time.time() * 1000))
                dynamodb_client = boto3.client('dynamodb', aws_access_key_id=os.getenv("ddb_ak"),
                                               aws_secret_access_key=os.getenv("ddb_sk"),
                                               region_name='us-east-1')
                dynamodb_client.update_item(
                    TableName=os.getenv("session_table"),
                    Key={
                        'sessionID': {'S': session_id}  # 指定需要更新的主键
                    },
                    UpdateExpression='SET lastUpdateTimestamp = :val',
                    ExpressionAttributeValues={
                        ':val': {'N': str(sent_message_timestamp)}  # 更新的值，数字需要转换为字符串
                    },
                    # ReturnValues="UPDATED_NEW"  # 可以指定返回值选项
                )
                # 以下记录是将用户消息和AI消息记录到chat history表里，并且记录停止原因
                if chunk['delta']['stop_reason'] == 'end_turn':
                    message_item = {
                        'sessionID': session_id,
                        'receivedMessageTimestamp': received_message_timestamp,
                        'sentMessageTimestamp': sent_message_timestamp,
                        'userMessage': input_text,
                        'assistantMessage': assistant_message,
                        'stopReason': "end_turn"
                    }
                    chat_history_table.put_item(Item=message_item)

                elif chunk['delta']['stop_reason'] == 'max_tokens':
                    message_item = {
                        'sessionID': session_id,
                        'receivedMessageTimestamp': received_message_timestamp,
                        'sentMessageTimestamp': sent_message_timestamp,
                        'userMessage': input_text,
                        'assistantMessage': assistant_message,
                        'stopReason': "max_tokens"
                    }
                    chat_history_table.put_item(Item=message_item)
                elif chunk['delta']['stop_reason'] == 'stop_sequence':
                    message_item = {
                        'sessionID': session_id,
                        'receivedMessageTimestamp': received_message_timestamp,
                        'sentMessageTimestamp': sent_message_timestamp,
                        'userMessage': input_text,
                        'assistantMessage': assistant_message,
                        'stopReason': "stop_sequence"
                    }
                    chat_history_table.put_item(Item=message_item)


    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        raise HTTPException(status_code=500, detail=f"Client error: {message}")


@app.get("/sessions/")
async def get_sessions(user: str = Query(..., description="User ID to filter sessions"),
                       model: str = Query(..., description="Model name to filter sessions")):
    # 使用 DynamoDB 查询session数据
    try:
        dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.getenv("ddb_ak"),
                                  aws_secret_access_key=os.getenv("ddb_sk"),
                                  region_name='us-east-1')
        table = dynamodb.Table(os.getenv("session_table"))
        response = table.query(
            IndexName='UserModelIndex',  # 这是已经创建的 GSI
            KeyConditionExpression=Key('user').eq(user) & Key('model').eq(model)
        )
        items = response['Items']
        # 根据 lastUpdateTimestamp 降序排序
        sorted_items = sorted(
            [item for item in items if int(item['lastUpdateTimestamp']) >= thirty_days_ago_timestamp],
            key=lambda x: x['lastUpdateTimestamp'],
            reverse=True
        )
        return sorted_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/")
async def delete_sessions(session_id: str = Query(..., description="Session ID to filter sessions")):
    # 删除某一个session
    try:
        dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.getenv("ddb_ak"),
                                  aws_secret_access_key=os.getenv("ddb_sk"),
                                  region_name='us-east-1')
        session_table = dynamodb.Table(os.getenv("session_table"))
        chat_history_table = dynamodb.Table(os.getenv("chat_history_table"))
        session_table.delete_item(
            Key={
                'sessionID': session_id  # 确保这里的键名和类型与表的分区键匹配
            }
        )
        response = chat_history_table.query(
            KeyConditionExpression='sessionID = :sid',
            ExpressionAttributeValues={':sid': session_id},
        )
        # 轮询chat history表，删除所有session id为传过来的值的item，还必须指定排序键，要不不能删
        for item in response.get('Items', []):
            chat_history_table.delete_item(
                Key={
                    'sessionID': item['sessionID'],
                    'receivedMessageTimestamp': item['receivedMessageTimestamp']
                }
            )
        # 删除成功回success，前端用
        return {"status": "success"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chathistory/")
async def get_sessions(session_id: str = Query(..., description="Session ID to filter sessions")):
    # 使用 DynamoDB 查询聊天历史
    try:
        chat_history = []
        dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.getenv("ddb_ak"),
                                  aws_secret_access_key=os.getenv("ddb_sk"),
                                  region_name='us-east-1')
        table = dynamodb.Table(os.getenv("chat_history_table"))
        response = table.query(
            KeyConditionExpression='sessionID = :sid',
            ExpressionAttributeValues={':sid': session_id},
            ScanIndexForward=True  # True 表示按排序键 receivedMessageTimestamp 升序排列
        )
        for item in response.get('Items', []):
            # 先放用户消息，再放AI消息
            user_message = {"sender": "You", "texts": [item.get("userMessage", "")]}
            chat_history.append(user_message)
            assistant_message = {"sender": "AI", "texts": [item.get("assistantMessage", "")]}
            chat_history.append(assistant_message)
        return chat_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
