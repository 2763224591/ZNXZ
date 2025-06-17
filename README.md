# ZNXZ
这里是ZNXZ的数据库端

向量切割的模型shibing624/text2vec-base-chinese

*模型太大没push

接口定义如下：

POST /update: 启动知识库更新

示例
>curl -X POST http://localhost:9000/update

GET /update-status: 查看更新状态

示例
>curl -X GET http://localhost:9000/update-status

POST /ask: 进行提问

示例
>curl -X POST -H "Content-Type: application/json" -d "{\"question\":\"软件工程的定义是什么？\"}" http://localhost:9000/ask

GET /status: 查看系统基本状态
