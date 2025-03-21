import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('printer_etl')

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "printer_management"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_sqlalchemy_engine():
    return create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

# Extract printer data
def extract_printer_data():
    logger.info("Extracting printer data from MySQL...")
    conn = get_db_connection()
    query = "SELECT * FROM PRINTERS"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Feature Engineering
def generate_features(df):
    df['last_maintenance'] = pd.to_datetime(df['last_maintenance'])
    df['days_since_last_maintenance'] = (datetime.now() - df['last_maintenance']).dt.days
    return df[['printer_id', 'total_page_count', 'days_since_last_maintenance']]

# Train Machine Learning Model
def train_model(df):
    logger.info("Training maintenance prediction model...")
    X = df[['total_page_count', 'days_since_last_maintenance']]
    y = df['days_since_last_maintenance'] * 1.2  # Simulated target variable
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Predict Maintenance
def predict_next_maintenance(model, df):
    predictions = []
    for _, row in df.iterrows():
        features = [[row['total_page_count'], row['days_since_last_maintenance']]]
        days_prediction = model.predict(features)[0]
        # print(days_prediction)
        predicted_days_remaining = max(0, days_prediction - row['days_since_last_maintenance'])
        predicted_date = datetime.now().date() + timedelta(days=predicted_days_remaining)
        # print(predicted_days_remaining)
        urgency = "URGENT" if predicted_days_remaining <= 7 else "Soon" if predicted_days_remaining <= 30 else "Normal"
        predictions.append({
            'printer_id': row['printer_id'],
            'predicted_maintenance_date': predicted_date,
            'predicted_days_until_maintenance': round(predicted_days_remaining),
            'urgency': urgency
        })
    return pd.DataFrame(predictions)

# Export to Power BI
def export_predictions_to_powerbi(predictions):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "maintenance_predictions1.csv")
    predictions.to_csv(file_path, index=False)
    logger.info(f"Predictions exported to {file_path}")
    return file_path

def save_predictions_to_db(predictions):
    engine = get_sqlalchemy_engine()
    predictions.to_sql("printer_maintenance_predictions", engine, if_exists="replace", index=False)
    logger.info("Predictions saved to database for Power BI visualization.")

# Run ETL and ML Process
def run_etl_pipeline():
    try:
        df = extract_printer_data()
        features_df = generate_features(df)
        model = train_model(features_df)
        predictions = predict_next_maintenance(model, features_df)
        export_predictions_to_powerbi(predictions)
        save_predictions_to_db(predictions)
        logger.info("ETL pipeline completed successfully.")
    except Exception as e:
        logger.error(f"ETL process failed: {e}")

if __name__ == "__main__":
    run_etl_pipeline()
