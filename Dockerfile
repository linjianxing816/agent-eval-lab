# 最小可复现环境：别人 clone 下来能跑通测试就够了。
# 不做多阶段构建、不写 docker-compose、不接 CI——
# 个人项目里这三样都是空转（没有别人给你提 PR，也没有部署目标）。
FROM python:3.11-slim

WORKDIR /app

# 先只拷依赖清单：改代码不会让这层缓存失效
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 测试全是离线的（FakeClient / MockTransport），不需要 API key
CMD ["python", "-m", "pytest", "tests", "-q"]
