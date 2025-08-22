from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.llms import Ollama

from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
import sys

db_user = "ts_user"
db_pass = "ts_pass"
db_host = "localhost"
db_port = 5432
db_name = "ticks"

db = SQLDatabase.from_uri(
f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
)

# Use the local LLaMA 3 model
llm = Ollama(model="llama3")

agent_executor = create_sql_agent(
llm=llm,
db=db,
agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
verbose=True
)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py '<natural language query>'")
        sys.exit(1)

    nl_query = sys.argv[1]
    result = agent_executor.run(nl_query)
    print(result)


