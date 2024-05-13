# 使用官方 Python 3.9 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将依赖信息复制到容器内
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件到容器内的工作目录
COPY . .

# 暴露容器的8000端口
EXPOSE 8000

# 启动应用程序
#CMD ["uvicorn", "llmchatbotapi:app", "--host", "0.0.0.0", "--port", "8000","--reload"]
CMD ["python", "llmchatbotapi.py"]