from typing import List, Dict, Any
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    insert,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from transformers.agents import Tool, ReactCodeAgent, HfEngine
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database(db_url: str = "sqlite:///:memory:") -> Engine:
    """Create a database engine."""
    try:
        engine = create_engine(db_url)
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Error creating database engine: {e}")
        raise

def create_table(engine: Engine, table_name: str, columns: List[Column]) -> Table:
    """Create a table in the database."""
    metadata_obj = MetaData()
    table = Table(table_name, metadata_obj, *columns)
    try:
        metadata_obj.create_all(engine)
        return table
    except SQLAlchemyError as e:
        logger.error(f"Error creating table: {e}")
        raise

def insert_data(engine: Engine, table: Table, rows: List[Dict[str, Any]]) -> None:
    """Insert data into the table."""
    try:
        with engine.begin() as connection:
            for row in rows:
                stmt = insert(table).values(**row)
                connection.execute(stmt)
    except SQLAlchemyError as e:
        logger.error(f"Error inserting data: {e}")
        raise

def get_table_description(engine: Engine, table_name: str) -> str:
    """Get the description of the table."""
    inspector = inspect(engine)
    columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table_name)]
    return "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])

class SQLExecutorTool(Tool):
    """SQL Executor Tool for performing SQL queries."""

    name = "sql_engine"
    description = "Allows you to perform SQL queries on the table. Returns a string representation of the result."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be correct SQL.",
        }
    }
    output_type = "text"

    def __init__(self, engine: Engine, table_name: str):
        super().__init__()
        self.engine = engine
        self.table_name = table_name
        self.description += f"\nThe table is named '{table_name}'. Its description is as follows: \n{get_table_description(engine, table_name)}"

    def forward(self, query: str) -> str:
        try:
            with self.engine.connect() as con:
                rows = con.execute(text(query))
                return "\n".join(str(row) for row in rows)
        except SQLAlchemyError as e:
            logger.error(f"Error executing SQL query: {e}")
            return f"Error: {e}"

def create_agent(engine: Engine, table_name: str, model_name: str) -> ReactCodeAgent:
    """Create a ReactCodeAgent with SQLExecutorTool."""
    sql_tool = SQLExecutorTool(engine, table_name)
    return ReactCodeAgent(
        tools=[sql_tool],
        llm_engine=HfEngine(model_name),
    )


def main():
    # Create database and table
    engine = create_database()
    table = create_table(
        engine,
        "receipts",
        [
            Column("receipt_id", Integer, primary_key=True),
            Column("customer_name", String(16), primary_key=True),
            Column("price", Float),
            Column("tip", Float),
        ]
    )

    # Insert sample data
    rows = [
        {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
        {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
        {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
        {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
    ]
    insert_data(engine, table, rows)

    # Create agent
    agent = create_agent(engine, "receipts", "meta-llama/Meta-Llama-3-70B-Instruct")

    # Run agent
    query = "What is the total amount of tips?"
    result = agent.run(query)
    logger.info(f"Query: {query}")
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    main()