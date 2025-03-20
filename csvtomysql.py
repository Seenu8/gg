import pandas as pd
from sqlalchemy import create_engine

# Database connection details
db_user = "root"      # MySQL username
db_password = "root"  # MySQL password
db_host = "localhost" # MySQL server (localhost)
db_name = "printer_management" # Database name

# Create MySQL connection using SQLAlchemy
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Define file path (modify if necessary)
file_path = "C:/Users/HP ADMIN/PycharmProjects/PythonProject/print_jobs.csv"  # Your uploaded CSV file

# Load CSV into Pandas DataFrame
df = pd.read_csv(file_path)

# Insert Data into MySQL (Append to existing data)
try:
    df.to_sql(name="print_jobs", con=engine, if_exists="append", index=False)
    print("Data inserted successfully into PRINTERS table!")
except Exception as e:
    print(f"Error inserting data: {e}")
