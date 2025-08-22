import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_ollama import OllamaLLM
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.prompts import PromptTemplate
import mplfinance as mpf

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
    if sql_text.startswith("```sql"):
        sql_text = sql_text.split("```sql")[1]
    if sql_text.endswith("```"):
        sql_text = sql_text.split("```")[0]
    sql_text = sql_text.strip()
    # Remove LIMIT clause if present
    import re
    sql_text = re.sub(r'LIMIT\s+\d+\s*;?', '', sql_text, flags=re.IGNORECASE)
    return sql_text.strip()

# Function to plot candlestick chart for verification
def plot_candlestick(df, output_file="candlestick_chart.png"):
    if not all(col in df.columns for col in ['bucket', 'open', 'high', 'low', 'close']):
        print("Error: DataFrame must contain 'bucket', 'open', 'high', 'low', 'close' columns for plotting.")
        return
    
    # Prepare DataFrame for mplfinance
    df = df.copy()  # Avoid modifying original
    df['bucket'] = pd.to_datetime(df['bucket'])
    df.set_index('bucket', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']] if 'volume' in df.columns else df[['open', 'high', 'low', 'close']]
    
    # Plot and save candlestick chart
    mpf.plot(
        df,
        type='candle',
        style='charles',  # TradingView-like style
        title='Candlestick Chart for Verification',
        ylabel='Price',
        volume=True if 'volume' in df.columns else False,
        savefig=output_file
    )
    print(f"Candlestick chart saved to {output_file}")

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query TimescaleDB with NLP and optionally plot OHLC data")
    parser.add_argument("query", type=str, help="Natural language query")
    parser.add_argument("--plot", action="store_true", help="Generate a candlestick plot for verification")
    args = parser.parse_args(args)

    query = args.query
    print(f"Processing query: {query}")
    
    # Generate SQL from the natural language query
    sql = sql_chain.invoke({"question": query})  # Removed top_k
    sql = clean_sql(sql)
    print(f"Generated SQL: {sql}")
    
    # Execute SQL and return results as Pandas DataFrame
    try:
        df = pd.read_sql(sql, engine)
        # Convert 'bucket' to datetime if not already
        if 'bucket' in df.columns:
            df['bucket'] = pd.to_datetime(df['bucket'])
        print("\nDataFrame output:")
        print(df)
        
        # Plot if --plot flag is provided
        if args.plot:
            plot_candlestick(df)
        
        return df  # Return for potential service wrapper
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return

if __name__ == "__main__":
    main()