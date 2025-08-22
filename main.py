import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.prompts import PromptTemplate
import mplfinance as mpf
import re


# Setup LLM (Ollama with LLaMA3)
llm = OllamaLLM(model="llama3")

# Database connection (update URI if your setup differs)
DB_URI = "postgresql://ts_user:ts_pass@localhost:5432/ticks"
db = SQLDatabase.from_uri(DB_URI)
engine = create_engine(DB_URI)

# Custom prompt for ohlcv_1m table
SQL_PROMPT = PromptTemplate(
    input_variables=["input", "top_k", "table_info"],
    template="""You are a PostgreSQL expert. Given a natural language query about time-series financial data, generate a SQL query for a table named 'ohlcv_1m' with columns: bucket (timestamptz, start of 1-minute window), symbol (text), open (double precision), high (double precision), low (double precision), close (double precision), volume (double precision). The query should:
- Select bucket, open, high, low, close, volume.
- Filter by symbol (e.g., 'PLTR') when specified.
- Handle date ranges:
    - For a single date (e.g., '2025-08-07'), include data from 00:00:00 to 23:59:59.999 on that date.
    - For ranges like '5 days before and after', use PostgreSQL interval syntax with timestamptz.
- Order results by bucket ascending.
- Use the 'ticks' database.
- Limit to {top_k} rows if applicable.
- Assume the table contains 1-minute OHLC data, so no interval column exists.

Table info: {table_info}

Query: {input}

Generate the SQL query and return only the SQL (no explanations or markdown).
Never use parameters or placeholders like $1, $2, :symbol, or :date. Always substitute actual values for symbol and date directly in the SQL. For queries like "5 days before and after", use INTERVAL arithmetic with timestamptz, e.g.:
AND bucket >= '<date>'::timestamptz - INTERVAL '5 days'
AND bucket < '<date>'::timestamptz + INTERVAL '6 days'

Example input: "show me data for 2025-08-07 for PLTR"
Example output: 
SELECT bucket, open, high, low, close, volume 
FROM ohlcv_1m 
WHERE symbol = 'PLTR' 
AND bucket >= '2025-08-07'::timestamptz 
AND bucket < '2025-08-08'::timestamptz 
ORDER BY bucket;

Example input: "Return PLTR data 5 days before and after 2025-09-10 for 1 min candles"
Example output: 
SELECT bucket, open, high, low, close, volume 
FROM ohlcv_1m 
WHERE symbol = 'PLTR' 
AND bucket >= '2025-09-10'::timestamptz - INTERVAL '5 days' 
AND bucket < '2025-09-10'::timestamptz + INTERVAL '6 days' 
ORDER BY bucket;
"""
)

# Chain to generate SQL from natural language
sql_chain = create_sql_query_chain(llm, db, prompt=SQL_PROMPT)

# Function to clean up generated SQL (removes markdown/code blocks if present)
def clean_sql(sql_text):
    # Remove markdown/code blocks if present
    sql_text = re.sub(r"```sql|```", "", sql_text, flags=re.IGNORECASE).strip()
    # Remove any prefix before the actual SQL (e.g., "Generated SQL: ...")
    # Find the first occurrence of SELECT and take everything from there
    match = re.search(r"(SELECT[\s\S]+?;)", sql_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return the original text if no SELECT found
    return sql_text.strip()

def extract_symbol_from_sql(sql):
    # Match WHERE symbol = 'SYMBOL'
    match = re.search(r"WHERE\s+symbol\s*=\s*'([^']+)'", sql, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# Function to plot candlestick chart for verification
def plot_candlestick(df, symbol=None, description=None):
    
    # Prepare title and output file name
    chart_title = f"Candlestick Chart for {symbol}" if symbol else "Candlestick Chart"
    output_file = f"candlestick_chart_{symbol}.png" if symbol else "candlestick_chart.png"
    # Prepare description text
    desc_text = description if description else f"This chart represents 1-minute OHLCV time-series data for {symbol}."
    # Plot and annotate
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        title=chart_title,
        ylabel='Price',
        volume=True if 'volume' in df.columns else False,
        returnfig=True
    )
    fig.text(0.5, 0.01, desc_text, ha='center', fontsize=10)
    fig.savefig(output_file)
    print(f"{chart_title} saved to {output_file} with description.")

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query TimescaleDB with NLP and optionally plot OHLC data")
    parser.add_argument("query", type=str, help="Natural language query")
    parser.add_argument("--plot", action="store_true", help="Generate a candlestick plot for verification")
    args = parser.parse_args(args)

    query = args.query
    print(f"Processing query: {query}")
    
    # Generate SQL from the natural language query
    sql = sql_chain.invoke({"question": query}) 
    sql = clean_sql(sql)
    print(f"Generated SQL: {sql}")

    # Extract symbol from SQL
    symbol = extract_symbol_from_sql(sql)
    
    # Execute SQL and return results as Pandas DataFrame
    try:
        df = pd.read_sql(sql, engine)
        if 'bucket' in df.columns:
            df['bucket'] = pd.to_datetime(df['bucket'])
            df = df.set_index('bucket')
        print("\nDataFrame output:")
        print(df)
        
        # Plot if --plot flag is provided
        if args.plot:
            plot_candlestick(df, symbol=symbol, description=query)
        
        return df
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return

if __name__ == "__main__":
    main()