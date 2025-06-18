FROM python:3.10-slim

# 创建并设置 APT 源文件（使用 echo 重定向）
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb-src http://mirrors.aliyun.com/debian/ bookworm main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security/ bookworm-security main" >> /etc/apt/sources.list && \
    echo "deb-src http://mirrors.aliyun.com/debian-security/ bookworm-security main" >> /etc/apt/sources.list

# 继续剩余的 Dockerfile 命令
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装Python依赖（使用国内镜像+PyTorch官方源）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu

# 复制应用代码
COPY main.py ./
COPY llm_api.py ./
COPY QueryProcessor.py ./
COPY SystemConfig.py ./
COPY KnowledgeBaseManager.py ./
COPY GraphTransformer.py ./
COPY DataProcessor.py ./
COPY DatabaseManager.py ./
COPY web.html ./

# 创建必要的目录
RUN mkdir -p course_data logs models--shibing624--text2vec-base-chinese
COPY course_data ./course_data
COPY models--shibing624--text2vec-base-chinese ./models--shibing624--text2vec-base-chinese

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 9000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/status || exit 1

# 启动命令
CMD ["python", "main.py"]