import re



def extract_all_where_clauses(sql_query):
    """
    Extract all WHERE clauses from a SQL query, including the WHERE keyword and trailing terminators.
    
    Parameters:
    sql_query (str): The SQL query string.
    
    Returns:
    list: A list of WHERE clause conditions as strings, including the WHERE keyword and trailing terminators.
    """
    # Convert the query to uppercase for case-insensitive matching
    sql_query = sql_query.upper()

    # Regular expression to match WHERE clauses including trailing terminators
    where_pattern = r'WHERE\s+((?:[^;]*(?:GROUP BY|ORDER BY|HAVING|LIMIT|$))+)'
    matches = re.findall(where_pattern, sql_query, re.DOTALL | re.IGNORECASE)

    # Extract WHERE clauses
    where_clauses = [match.strip() for match in matches]

    # Clean up the WHERE clauses by removing comments and unnecessary whitespace
    cleaned_clauses = []
    for clause in where_clauses:
        # Remove comments
        clause = re.sub(r'--.*$', '', clause, flags=re.MULTILINE | re.IGNORECASE).strip()
        # Normalize whitespace
        clause = re.sub(r'\s+', ' ', clause)
        cleaned_clauses.append(clause)

    return cleaned_clauses

# Example usage
sql_query = """
WITH cte AS (
    SELECT * FROM orders WHERE order_date >= '2024-01-01'
)
SELECT * FROM customers
JOIN cte ON customers.id = cte.customer_id
WHERE customers.status = 'active' AND cte.amount > 1000
WHERE customers.age > 30
GROUP BY customers.status
ORDER BY cte.amount DESC
"""

# Extract WHERE clauses
where_clauses = extract_all_where_clauses(sql_query)

# Print results
print("WHERE Clauses:")
for idx, clause in enumerate(where_clauses, 1):
    print(f"Clause {idx}: {clause}")



import pandas as pd
import re

def read_csv_with_encoding(file_path, encoding='utf-8'):
    """
    Read a CSV file with the specified encoding.
    
    Parameters:
    file_path (str): Path to the CSV file.
    encoding (str): Encoding of the file.
    
    Returns:
    DataFrame: Contents of the CSV file.
    """
    return pd.read_csv(file_path, encoding=encoding, errors='replace')

def extract_all_where_clauses(sql_query):
    """
    Extract all WHERE clauses from a SQL query, including the WHERE keyword and trailing terminators.
    
    Parameters:
    sql_query (str): The SQL query string.
    
    Returns:
    list: A list of WHERE clause conditions as strings, including the WHERE keyword and trailing terminators.
    """
    # Convert the query to uppercase for case-insensitive matching
    sql_query = sql_query.upper()

    # Regular expression to match WHERE clauses including trailing terminators
    where_pattern = r'WHERE\s+((?:[^;]*(?:GROUP BY|ORDER BY|HAVING|LIMIT|$))+)'
    matches = re.findall(where_pattern, sql_query, re.DOTALL | re.IGNORECASE)

    # Extract WHERE clauses
    where_clauses = [match.strip() for match in matches]

    # Clean up the WHERE clauses by removing comments and unnecessary whitespace
    cleaned_clauses = []
    for clause in where_clauses:
        # Remove comments
        clause = re.sub(r'--.*$', '', clause, flags=re.MULTILINE | re.IGNORECASE).strip()
        # Normalize whitespace
        clause = re.sub(r'\s+', ' ', clause)
        cleaned_clauses.append(clause)

    return cleaned_clauses

# Example usage
csv_file_path = 'your_data.csv'  # Replace with your actual CSV file path
encoding = 'latin1'  # Adjust encoding as needed

# Read SQL queries from CSV
data = read_csv_with_encoding(csv_file_path, encoding=encoding)

# Assuming SQL queries are in a column named 'SQL_query'
for idx, row in data.iterrows():
    sql_query = row['SQL_query']
    print(f"\nExtracting WHERE clauses from SQL query {idx+1}:")
    
    # Extract WHERE clauses
    where_clauses = extract_all_where_clauses(sql_query)
    
    # Print results
    for clause_idx, clause in enumerate(where_clauses, 1):
        print(f"Clause {clause_idx}: {clause}")

