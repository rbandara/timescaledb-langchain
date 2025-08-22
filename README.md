# 🚀 Tutorial: Querying TimescaleDB with LangChain + Ollama (Free & Local Setup)

If you want to showcase **AI + databases** without paying for API credits, you can combine **LangChain**, **TimescaleDB**, and **Ollama** to run natural language SQL queries locally. This tutorial walks through a step-by-step setup you can publish in your portfolio.

---

## 📌 What We’ll Build
We’ll create a system where you can:
- Connect to a **TimescaleDB** (PostgreSQL extension for time-series data).
- Run **LangChain SQL agent** powered by **Ollama** with the **LLaMA model**.
- Ask natural language questions like:
  - *“How many rows are in the `trades` hypertable?”*
  - *“Show me the average trade size per minute in the last hour.”*

All without an **OpenAI API key** or paid service.

---

## 1️⃣ Prerequisites
- **Python 3.9+**
- **TimescaleDB** running locally or in Docker
- **Ollama** installed ([Download here](https://ollama.com/))
- LLaMA model pulled:
  ```bash
  ollama pull llama3
  ```

---

## 2️⃣ Install Dependencies
```bash
pip install langchain langchain-community psycopg2-binary
```

---

## 3️⃣ Connect LangChain to TimescaleDB
```python
from langchain_community.utilities.sql_database import SQLDatabase

db_user = "ts_user"
db_pass = "ts_pass"
db_host = "localhost"
db_port = 5432
db_name = "ticks"

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
)

print("Dialect:", db.dialect)
print("Tables:", db.get_usable_table_names())
```

✅ This verifies your connection and lists usable tables.

---

## 4️⃣ Add Ollama as the LLM
```python
from langchain_community.llms import Ollama

# Use the local LLaMA 3 model
llm = Ollama(model="llama3")
```

---

## 5️⃣ Create a SQL Agent
```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

---

## 6️⃣ Ask Questions in Plain English 🎉
```python
query = "How many rows are in the trades hypertable?"
result = agent_executor.run(query)
print(result)

query = "Show me the average trade size per minute in the last hour."
result = agent_executor.run(query)
print(result)
```

Behind the scenes:
- LangChain converts your **natural language** to SQL.
- TimescaleDB executes the query.
- Ollama interprets results with LLaMA.

---

## 7️⃣ Bonus: TimescaleDB Functions
TimescaleDB offers time-series SQL functions like `time_bucket`.

Example query (agent should be able to generate this):
```sql
SELECT time_bucket('1 minute', time) AS bucket,
       AVG(size) AS avg_trade_size
FROM trades
WHERE time > now() - interval '1 hour'
GROUP BY bucket
ORDER BY bucket;
```

This makes it easy to query financial tick data, IoT metrics, or any time-series dataset.

---

## ✅ Summary
You now have:
- **TimescaleDB** for time-series storage.
- **LangChain SQL agent** to convert questions → SQL.
- **Ollama with LLaMA** running locally (free, no API key needed).


---

### 🔮 Next Steps
- Swap Ollama with **Groq** or **Together AI** for faster hosted models.
- Add a **Streamlit UI** to make an interactive demo.
- Extend queries with **Matplotlib** to plot results (candlestick charts, etc.).

