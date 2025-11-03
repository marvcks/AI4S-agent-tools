from langchain_openai import ChatOpenAI
import os

# chatLLM = ChatOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     # other params...
# )

# chatLLM =  ChatOpenAI(
#                 model="gpt-4o-mini",
#                 base_url="https://openai.weavex.tech/v1"
#                 )
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "你是谁？"}]
# response = chatLLM.invoke(messages)
# print(response.json())

# from langchain_community.embeddings import DashScopeEmbeddings
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v2",
#     # other params...
# )
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://openai.weavex.tech/v1/"
    )

input_text = "The meaning of life is 42"
vector = embeddings.embed_query("hello")
print(vector[:3])