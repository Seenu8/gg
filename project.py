import pandas as pd
import numpy as np
import pymysql
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine


class PrinterPredictiveMaintenance:
    """
    A system that predicts when printers will need maintenance based on
    historical data and current usage patterns.
    """

    def __init__(self, db_config):
        """
        Initialize the predictive maintenance system with database configuration.

        Args:
            db_config (dict): Contains database connection parameters
                              (host, user, password, database)
        """
        self.db_config = db_config
        self.connection = None
        self.model = None
        self.feature_pipeline = None

    def connect_to_database(self):
        """Establish connection to the database using SQLAlchemy."""
        try:
            db_url = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
            self.engine = create_engine(db_url)  # Create an SQLAlchemy engine
            self.connection = self.engine.connect()  # Create a connection
            print("Database connection established successfully.")
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    def fetch_printer_data(self):
        """
        Fetch printer information from the database.

        Returns:
            pandas.DataFrame: Data containing printer information
        """
        query = """
        SELECT 
            p.printer_id,
            p.model,
            p.location,
            p.department_id,
            p.installation_date,
            p.last_maintenance,
            p.total_page_count,
            p.status,
            d.name as department_name
        FROM 
            PRINTERS p
        JOIN 
            DEPARTMENTS d ON p.department_id = d.department_id
        """

        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Error fetching printer data: {e}")
            raise

    def fetch_maintenance_records(self):
        """
        Fetch maintenance records from the database.

        Returns:
            pandas.DataFrame: Data containing maintenance records
        """
        query = """
        SELECT 
            m.record_id,
            m.printer_id,
            m.maintenance_date,
            m.technician,
            m.description,
            m.cost,
            p.model,
            p.department_id
        FROM 
            MAINTENANCE_RECORDS m
        JOIN 
            PRINTERS p ON m.printer_id = p.printer_id
        """

        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Error fetching maintenance records: {e}")
            raise

    def fetch_print_jobs(self):
        """
        Fetch print job information from the database.

        Returns:
            pandas.DataFrame: Data containing print job information
        """
        query = """
        SELECT 
            j.job_id,
            j.user_id,
            j.printer_id,
            j.print_time,
            j.page_count,
            j.duplex,
            j.color,
            j.paper_size,
            j.document_type,
            j.cost
        FROM 
            PRINT_JOBS j
        """

        try:
            return pd.read_sql(query, self.engine)
        except Exception as e:
            print(f"Error fetching print job data: {e}")
            raise

    def prepare_training_data(self):
        """
        Prepare data for training the predictive maintenance model.

        This function processes printer, maintenance, and print job data to create features
        that can predict maintenance needs.

        Returns:
            tuple: X (features) and y (days until next maintenance needed)
        """
        # Fetch data
        printers_df = self.fetch_printer_data()
        maintenance_df = self.fetch_maintenance_records()
        print_jobs_df = self.fetch_print_jobs()

        # Convert date columns to datetime
        printers_df['installation_date'] = pd.to_datetime(printers_df['installation_date'])
        printers_df['last_maintenance'] = pd.to_datetime(printers_df['last_maintenance'])
        maintenance_df['maintenance_date'] = pd.to_datetime(maintenance_df['maintenance_date'])
        print_jobs_df['print_time'] = pd.to_datetime(print_jobs_df['print_time'])

        # Create training dataset by looking at historical maintenance patterns
        maintenance_history = []

        # Group maintenance records by printer
        maintenance_by_printer = maintenance_df.sort_values('maintenance_date').groupby('printer_id')

        for printer_id, printer_maintenance in maintenance_by_printer:
            # Skip if there's only one maintenance record (we need at least two to calculate intervals)
            if len(printer_maintenance) < 2:
                continue

            # Calculate days between maintenance events
            printer_maintenance = printer_maintenance.sort_values('maintenance_date')
            printer_maintenance['next_maintenance_date'] = printer_maintenance['maintenance_date'].shift(-1)
            printer_maintenance['days_until_next_maintenance'] = (
                    printer_maintenance['next_maintenance_date'] - printer_maintenance['maintenance_date']
            ).dt.days

            # Drop the last row (which has no next_maintenance_date)
            printer_maintenance = printer_maintenance.dropna(subset=['days_until_next_maintenance'])

            # Add printer information at the time of maintenance
            for idx, maint_record in printer_maintenance.iterrows():
                # Get printer info at the time of this maintenance
                printer_info = printers_df[printers_df['printer_id'] == printer_id].iloc[0].copy()

                # Get age of printer at maintenance time (in days)
                printer_age_days = (maint_record['maintenance_date'] - printer_info['installation_date']).days

                # Get print jobs since last maintenance or installation
                last_date = printer_maintenance[
                    printer_maintenance['maintenance_date'] < maint_record['maintenance_date']]
                if not last_date.empty:
                    last_date = last_date['maintenance_date'].max()
                else:
                    last_date = printer_info['installation_date']

                jobs_since_last = print_jobs_df[
                    (print_jobs_df['printer_id'] == printer_id) &
                    (print_jobs_df['print_time'] > last_date) &
                    (print_jobs_df['print_time'] <= maint_record['maintenance_date'])
                    ]

                # Calculate job-related features
                total_pages = jobs_since_last['page_count'].sum() if not jobs_since_last.empty else 0
                color_jobs_pct = jobs_since_last['color'].mean() * 100 if not jobs_since_last.empty else 0
                duplex_jobs_pct = jobs_since_last['duplex'].mean() * 100 if not jobs_since_last.empty else 0

                # Count jobs by document type
                doc_types = jobs_since_last[
                    'document_type'].value_counts().to_dict() if not jobs_since_last.empty else {}

                # Create a record with all features
                record = {
                    'printer_id': printer_id,
                    'model': printer_info['model'],
                    'department_id': printer_info['department_id'],
                    'printer_age_days': printer_age_days,
                    'days_since_last_maintenance': (maint_record['maintenance_date'] - last_date).days,
                    'pages_printed_since_last': total_pages,
                    'color_jobs_percentage': color_jobs_pct,
                    'duplex_jobs_percentage': duplex_jobs_pct,
                    'maintenance_cost': maint_record['cost'],
                    'days_until_next_maintenance': maint_record['days_until_next_maintenance']
                }

                # Add document type counts
                for doc_type in ['report', 'presentation', 'form', 'blueprint', 'technical_spec', 'contract',
                                 'brochure']:
                    record[f'doc_type_{doc_type}'] = doc_types.get(doc_type, 0)

                maintenance_history.append(record)

        # Convert to DataFrame
        training_df = pd.DataFrame(maintenance_history)

        if training_df.empty:
            raise ValueError("Not enough maintenance history to build a predictive model")

        # Separate features and target
        X = training_df.drop(['days_until_next_maintenance'], axis=1)
        y = training_df['days_until_next_maintenance']

        return X, y

    def build_model(self):
        """
        Build and train the predictive maintenance model.

        Returns:
            sklearn.pipeline.Pipeline: Trained model pipeline
        """
        X, y = self.prepare_training_data()

        if len(X) < 2:
            raise ValueError("Not enough data for train-test split. Need at least 2 records.")

        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Create preprocessor for features
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', 'passthrough', numeric_cols)
            ]
        )

        # Create the model pipeline
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Train the model
        model_pipeline.fit(X_train, y_train)

        # Evaluate the model
        train_score = model_pipeline.score(X_train, y_train)
        test_score = model_pipeline.score(X_test, y_test)

        print(f"Model R² score on training data: {train_score:.4f}")
        print(f"Model R² score on test data: {test_score:.4f}")

        # Store the model and feature pipeline
        self.model = model_pipeline
        self.feature_columns = X.columns.tolist()

        return model_pipeline

    def generate_printer_features(self, printer_id):
        """
        Generate current features for a specific printer to predict its maintenance needs.

        Args:
            printer_id (str): The ID of the printer

        Returns:
            pandas.DataFrame: Features for the specified printer
        """
        # Fetch printer information
        printers_df = self.fetch_printer_data()
        printer_info = printers_df[printers_df['printer_id'] == printer_id].iloc[0]

        # Fetch maintenance records
        maintenance_df = self.fetch_maintenance_records()
        printer_maintenance = maintenance_df[maintenance_df['printer_id'] == printer_id].sort_values('maintenance_date')

        # Fetch print jobs
        print_jobs_df = self.fetch_print_jobs()

        # Calculate current printer age in days
        current_date = datetime.now().date()
        printer_age_days = (current_date - printer_info['installation_date'].date()).days

        # Get the last maintenance date
        last_maintenance_date = printer_info['last_maintenance'].date()
        days_since_last_maintenance = (current_date - last_maintenance_date).days

        # Get print jobs since last maintenance
        jobs_since_last = print_jobs_df[
            (print_jobs_df['printer_id'] == printer_id) &
            (print_jobs_df['print_time'] > pd.Timestamp(last_maintenance_date))
            ]

        # Calculate job-related features
        total_pages = jobs_since_last['page_count'].sum() if not jobs_since_last.empty else 0
        color_jobs_pct = jobs_since_last['color'].mean() * 100 if not jobs_since_last.empty else 0
        duplex_jobs_pct = jobs_since_last['duplex'].mean() * 100 if not jobs_since_last.empty else 0

        # Count jobs by document type
        doc_types = jobs_since_last['document_type'].value_counts().to_dict() if not jobs_since_last.empty else {}

        # Last maintenance cost (use average if not available)
        if not printer_maintenance.empty:
            last_maintenance_cost = printer_maintenance.iloc[-1]['cost']
        else:
            # Use average maintenance cost across all printers
            last_maintenance_cost = maintenance_df['cost'].mean()

        # Create a features record
        features = {
            'printer_id': printer_id,
            'model': printer_info['model'],
            'department_id': printer_info['department_id'],
            'printer_age_days': printer_age_days,
            'days_since_last_maintenance': days_since_last_maintenance,
            'pages_printed_since_last': total_pages,
            'color_jobs_percentage': color_jobs_pct,
            'duplex_jobs_percentage': duplex_jobs_pct,
            'maintenance_cost': last_maintenance_cost
        }

        # Add document type counts
        for doc_type in ['report', 'presentation', 'form', 'blueprint', 'technical_spec', 'contract', 'brochure']:
            features[f'doc_type_{doc_type}'] = doc_types.get(doc_type, 0)

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # Ensure all expected features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        return features_df[self.feature_columns]

    def predict_next_maintenance(self, printer_id=None):
        """
        Predict when each printer will need maintenance next.

        Args:
            printer_id (str, optional): If provided, predictions will be made only for this printer.
                                       If None, predictions will be made for all printers.

        Returns:
            pandas.DataFrame: Predictions for each printer
        """
        if self.model is None:
            self.build_model()

        printers_df = self.fetch_printer_data()

        if printer_id:
            printers_to_predict = printers_df[printers_df['printer_id'] == printer_id]
        else:
            printers_to_predict = printers_df

        predictions = []

        for idx, printer in printers_to_predict.iterrows():
            printer_id = printer['printer_id']

            try:
                # Debugging: Print the type of last_maintenance
                print(
                    f"Printer ID: {printer_id}, Last Maintenance: {printer['last_maintenance']} ({type(printer['last_maintenance'])})")

                # Get last_maintenance from DataFrame
                last_maintenance = printer['last_maintenance']

                # Convert only if it's a string
                if isinstance(last_maintenance, str):
                    last_maintenance = datetime.strptime(last_maintenance, "%Y-%m-%d").date()

                # Convert only if it's a datetime (not date)
                elif isinstance(last_maintenance, datetime):
                    last_maintenance = last_maintenance.date()

                # 🚀 No conversion needed if it's already a `date` object

                # Calculate predicted maintenance date
                days_since_last = (datetime.now().date() - last_maintenance).days
                predicted_days_remaining = max(0,days_since_last)
                predicted_date = datetime.now().date() + timedelta(days=predicted_days_remaining)

                predictions.append({
                    'printer_id': printer_id,
                    'last_maintenance': last_maintenance,
                    'days_since_last_maintenance': days_since_last,
                    'predicted_maintenance_date': predicted_date
                })

            except Exception as e:
                print(f"Error predicting maintenance for printer {printer_id}: {e}")

        return pd.DataFrame(predictions)

    def analyze_maintenance_factors(self):
        """
        Analyze which factors most influence maintenance needs.

        Returns:
            dict: Feature importance information
        """
        if self.model is None:
            self.build_model()

        # Extract feature names from the model pipeline
        preprocessor = self.model.named_steps['preprocessor']
        model = self.model.named_steps['model']

        # Get one-hot encoded column names
        cat_cols = preprocessor.transformers_[0][2]
        num_cols = preprocessor.transformers_[1][2]

        # One-hot encoded feature names
        encoder = preprocessor.transformers_[0][1]
        cat_features = encoder.get_feature_names_out(cat_cols)

        # All feature names
        feature_names = np.concatenate([cat_features, np.array(num_cols)])

        # Get feature importances
        importances = model.feature_importances_

        # Create a DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Calculate cumulative importance
        feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()

        return {
            'importance_df': feature_importance,
            'top_features': feature_importance.head(10)['Feature'].tolist(),
            'top_importance': feature_importance.head(10)['Importance'].tolist()
        }

    def visualize_predictions(self, predictions):
        """
        Create visualizations for maintenance predictions.

        Args:
            predictions (pandas.DataFrame): Maintenance predictions

        Returns:
            dict: Matplotlib figures
        """
        figures = {}

        # Set the style
        plt.style.use('ggplot')

        # 1. Bar chart of days until next maintenance by printer
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        predictions_sorted = predictions.sort_values('predicted_days_until_maintenance')
        bars = ax1.bar(
            predictions_sorted['printer_id'],
            predictions_sorted['predicted_days_until_maintenance'],
            color=predictions_sorted['urgency'].map({
                'URGENT': 'red',
                'Soon': 'orange',
                'Normal': 'green'
            })
        )
        ax1.set_title('Predicted Days Until Next Maintenance', fontsize=16)
        ax1.set_xlabel('Printer ID', fontsize=12)
        ax1.set_ylabel('Days', fontsize=12)
        ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.7)
        ax1.axhline(y=7, color='red', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)

        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 1,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10
            )

        figures['days_until_maintenance'] = fig1

        # 2. Timeline of upcoming maintenance
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        # Create a timeline for the next 90 days
        today = datetime.now().date()
        dates = [today + timedelta(days=i) for i in range(90)]

        # For each printer, plot its maintenance date
        for i, (idx, printer) in enumerate(predictions.iterrows()):
            maint_date = printer['predicted_maintenance_date']
            if maint_date <= today + timedelta(days=90):
                days_from_now = (maint_date - today).days
                color = 'red' if printer['urgency'] == 'URGENT' else 'orange' if printer[
                                                                                     'urgency'] == 'Soon' else 'green'
                ax2.scatter(days_from_now, i, color=color, s=100, edgecolor='black')
                ax2.text(days_from_now + 1, i, f"{printer['printer_id']} - {printer['location']}", fontsize=10)

        ax2.set_yticks([])
        ax2.set_title('Maintenance Timeline (Next 90 Days)', fontsize=16)
        ax2.set_xlabel('Days from Today', fontsize=12)

        # Add reference lines for weeks
        for week in range(1, 13):
            ax2.axvline(x=week * 7, color='gray', linestyle='--', alpha=0.5)

        figures['maintenance_timeline'] = fig2

        # 3. Feature importance visualization
        factors = self.analyze_maintenance_factors()
        top_features = factors['top_features']
        top_importance = factors['top_importance']

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        bars = ax3.barh(top_features[-1::-1], top_importance[-1::-1])
        ax3.set_title('Top Factors Influencing Maintenance Needs', fontsize=16)
        ax3.set_xlabel('Relative Importance', fontsize=12)

        # Clean up feature names for display
        labels = [label.split('__')[1] if '__' in label else label for label in top_features[-1::-1]]
        ax3.set_yticklabels(labels)

        figures['feature_importance'] = fig3

        return figures

    def generate_maintenance_schedule(self):
        """
        Generate a maintenance schedule for all printers.

        Returns:
            pandas.DataFrame: Maintenance schedule
        """
        predictions = self.predict_next_maintenance()

        # Group printers by month for scheduling
        current_date = datetime.now().date()
        schedule = []

        # Group by month for the next 6 months
        for i in range(6):
            month_start = current_date + relativedelta(months=i, day=1)
            month_end = current_date + relativedelta(months=i + 1, day=1, days=-1)

            # Find printers that need maintenance this month
            month_printers = predictions[
                (predictions['predicted_maintenance_date'] >= month_start) &
                (predictions['predicted_maintenance_date'] <= month_end)
                ]

            if not month_printers.empty:
                # Sort by urgency and predicted date
                month_printers = month_printers.sort_values(['urgency', 'predicted_maintenance_date'])

                # Add to schedule with suggested week
                for j, (idx, printer) in enumerate(month_printers.iterrows()):
                    week_of_month = min(4, j // 3 + 1)  # Distribute across 4 weeks, max 3 printers per week
                    schedule.append({
                        'printer_id': printer['printer_id'],
                        'model': printer['model'],
                        'location': printer['location'],
                        'department_id': printer['department_id'],
                        'predicted_date': printer['predicted_maintenance_date'],
                        'urgency': printer['urgency'],
                        'scheduled_month': month_start.strftime('%B %Y'),
                        'suggested_week': f"Week {week_of_month}"
                    })

        return pd.DataFrame(schedule)

    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def run_full_analysis(self):
        """
        Run a complete analysis and return all results.

        Returns:
            dict: Complete analysis results
        """
        try:
            # Connect to database
            self.connect_to_database()

            # Build the model
            self.build_model()

            # Generate predictions for all printers
            predictions = self.predict_next_maintenance()

            # Create visualizations
            figures = self.visualize_predictions(predictions)

            # Generate maintenance schedule
            schedule = self.generate_maintenance_schedule()

            # Analyze factors influencing maintenance
            factors = self.analyze_maintenance_factors()

            # Close connection
            self.close_connection()

            return {
                'predictions': predictions,
                'schedule': schedule,
                'factors': factors,
                'figures': figures
            }
        except Exception as e:
            print(f"Error running analysis: {e}")
            if self.connection:
                self.close_connection()
            raise


# Example usage
def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'printer_management'
    }

    # Create the predictive maintenance system
    ppm = PrinterPredictiveMaintenance(db_config)

    # Run the full analysis
    try:
        analysis = ppm.run_full_analysis()

        # Print urgent maintenance needs
        urgent_printers = analysis['predictions'][analysis['predictions']['urgency'] == 'URGENT']
        print("\nURGENT MAINTENANCE NEEDED:")
        print(urgent_printers[['printer_id', 'location', 'predicted_maintenance_date']])

        # Print next month's schedule
        next_month = datetime.now().date() + relativedelta(months=1)
        next_month_name = next_month.strftime('%B %Y')
        next_month_schedule = analysis['schedule'][analysis['schedule']['scheduled_month'] == next_month_name]
        print(f"\nMAINTENANCE SCHEDULE FOR {next_month_name}:")
        print(next_month_schedule[['printer_id', 'location', 'suggested_week']])

        # Print top factors
        print("\nTOP FACTORS AFFECTING PRINTER MAINTENANCE:")
        for i, (feature, importance) in enumerate(zip(
                analysis['factors']['top_features'][:5],
                analysis['factors']['top_importance'][:5]
        )):
            feature_name = feature.split('__')[1] if '__' in feature else feature
            print(f"{i + 1}. {feature_name}: {importance:.4f}")

        # Save figures
        for name, fig in analysis['figures'].items():
            fig.savefig(f"printer_maintenance_{name}.png", dpi=300, bbox_inches='tight')
            print(f"Saved figure: printer_maintenance_{name}.png")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()