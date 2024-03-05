1. Create your virtual environment
2. Install from requirements
```$ pip istall -r requirements.txt```
3. Add your Javascript or Python files (defaults to /docs) you can add whatever you want, it will recursively digests files in the named directory.
4. Add your env file or use the basic template: ```$ cp example.env .env```
5. Add your api keys to the .env
6. If you do not have a qdrant store you can run it locally in memory (free cluster is available by signing up online - get your url and api keys once you do)
7. If you are running local in memory init the ragtool with ```qdrant_client.QdrantClient(":memory:")``` I think : / have not tested yet.
8. Once you have your qdrant and llm environment set and dependencies installed you should be able to add your prompt and run the app.py

I am working on cleaning it all up so its easier to switch files types and ultimately do it automatically. It uses the BGE small, try large if your computer is stronger than mine.
Once you have run the pipeline and loaded the documents you can use the alternative code to get_[language]_pipeline. This connects to the database instead of re-embedding everything. If you are using qdrant in memory this won't work unless you add a cache.

Have fun!
