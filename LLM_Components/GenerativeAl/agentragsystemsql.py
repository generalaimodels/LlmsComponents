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
from typing import List, Dict, Any
from transformers.agents import Tool, ReactCodeAgent, HfEngine

def create_receipts_table(engine: Engine, metadata: MetaData) -> Table:
    """Creates the receipts table in the database.

    Args:
        engine (Engine): The SQLAlchemy engine connected to the database.
        metadata (MetaData): The metadata object containing the table definitions.

    Returns:
        Table: The created receipts table.
    """
    table = Table(
        "receipts",
        metadata,
        Column("receipt_id", Integer, primary_key=True),
        Column("customer_name", String(16), primary_key=True),
        Column("price", Float),
        Column("tip", Float),
    )
    metadata.create_all(engine)
    return table

def insert_rows(engine: Engine, table: Table, rows: List[Dict[str, Any]]) -> None:
    """Inserts rows into the given table in the database.

    Args:
        engine (Engine): The SQLAlchemy engine connected to the database.
        table (Table): The table where rows will be inserted.
        rows (List[Dict[str, Any]]): The rows to be inserted.
    """
    with engine.begin() as connection:
        for row in rows:
            stmt = insert(table).values(**row)
            connection.execute(stmt)

def get_table_description(engine: Engine, table_name: str) -> str:
    """Generates a description of the table's columns.

    Args:
        engine (Engine): The SQLAlchemy engine connected to the database.
        table_name (str): The name of the table.

    Returns:
        str: The formatted description of the table's columns.
    """
    inspector = inspect(engine)
    columns_info = [(col["name"], col["type"]) for col in inspector.get_columns(table_name)]
    return "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])

class SQLExecutorTool(Tool):
    name = "sql_engine"
    description = "Allows you to perform SQL queries on the table. Returns a string representation of the result."

    def __init__(self, engine: Engine, table_desc: str) -> None:
        self.engine = engine
        self.description += f"\nThe table is named 'receipts'. Its description is as follows:\n{table_desc}"
        super().__init__()

    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be correct SQL.",
        }
    }
    output_type = "text"

    def forward(self, query: str) -> str:
        """Executes the given SQL query on the database.

        Args:
            query (str): The SQL query to execute.

        Returns:
            str: The string representation of the query result.
        """
        output = ""
        with self.engine.connect() as con:
            result = con.execute(text(query))
            for row in result:
                output += "\n" + str(row)
        return output

def main() -> None:
    """Main function to initialize the database, create the table, insert data, and run the SQL agent."""
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()

    receipts_table = create_receipts_table(engine, metadata)

    rows = [
        {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
        {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
        {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
        {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
    ]
    insert_rows(engine, receipts_table, rows)

    table_description = get_table_description(engine, "receipts")

    sql_tool = SQLExecutorTool(engine, table_description)

    agent = ReactCodeAgent(
        tools=[sql_tool],
        llm_engine=HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")
    )

    query = "SELECT * FROM receipts"
    result = agent.run(query)
    print(result)

if __name__ == "__main__":
    main()