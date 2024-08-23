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
