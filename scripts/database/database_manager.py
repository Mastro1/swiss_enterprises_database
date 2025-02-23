import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional
import logging
import re

class DatabaseManager:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def merge_local_ch_data(self, 
                           local_ch_file: str = 'local_ch_company_data.csv',
                           query_results_file: str = 'estratto_results.csv') -> pd.DataFrame:
        """
        Merge local.ch data with query results and filter for companies with contact info
        """
        try:
            # Read CSVs
            local_ch_df = pd.read_csv(local_ch_file)
            query_df = pd.read_csv(query_results_file)

            # Merge on IDI
            merged_df = pd.merge(query_df, local_ch_df, 
                               left_on='IDI', 
                               right_on='IDI', 
                               how='left')
            
            # Keep only rows with at least one contact info
            has_contact = (
                merged_df['Phone Numbers'].notna() | 
                merged_df['Email'].notna() | 
                merged_df['Website'].notna()
            )
            filtered_df = merged_df[has_contact]

            self.logger.info(f"Initial records: {len(query_df)}")
            self.logger.info(f"Records with contact info: {len(filtered_df)}")
            
            return filtered_df

        except Exception as e:
            self.logger.error(f"Error merging local.ch data: {str(e)}")
            raise

    def merge_with_classifications(self, 
                                 contact_df: pd.DataFrame,
                                 classified_file: str = 'classified_companies_ml.csv') -> pd.DataFrame:
        """
        Merge contact data with ML classifications
        """
        try:
            # Read classifications
            classified_df = pd.read_csv(classified_file)

            # Merge on IDI
            final_df = pd.merge(contact_df, 
                              classified_df[['IDI', 'Category', 'Subcategory']], 
                              on='IDI', 
                              how='left')

            self.logger.info(f"Final records with classifications: {len(final_df)}")
            self.logger.info("\nCategory distribution:")
            self.logger.info(final_df['Category'].value_counts())

            return final_df

        except Exception as e:
            self.logger.error(f"Error merging classifications: {str(e)}")
            raise

    def create_database(self, 
                       classified_file: str = 'classified_companies_openai_full.csv',
                       db_path: str = 'companies.db') -> None:
        """
        Create SQLite database with main and metadata tables from classified companies
        """
        try:
            # Read the classified data
            df = pd.read_csv(classified_file)
            
            # Print some raw dates to inspect
            print("\nSample of raw registration dates:")
            print(df['Registration_date'].value_counts().head(10))
            
            # Clean registration date with validation
            def clean_date(date_str):
                try:
                    if pd.isna(date_str):
                        return None
                        
                    # Extract date using regex
                    match = re.search(r'Registered on (\d{2}\.\d{2}\.\d{4})', str(date_str))
                    if not match:
                        print(f"No match for date format in: '{date_str}'")
                        return None
                        
                    date_str = match.group(1)
                    
                    # Print problematic dates for inspection
                    day, month, year = map(int, date_str.split('.'))
                    if year < 1800:  # Only filter very old dates
                        print(f"Too old date found: '{date_str}' -> D:{day} M:{month} Y:{year}")
                        return None
                        
                    if year > 2030:  # Allow up to 2030
                        print(f"Future date beyond 2030 found: '{date_str}' -> D:{day} M:{month} Y:{year}")
                        return None
                        
                    return pd.to_datetime(date_str, format='%d.%m.%Y')
                    
                except Exception as e:
                    print(f"Error processing date '{date_str}': {str(e)}")
                    return None
            
            # Apply date cleaning
            df['Registration_date'] = df['Registration_date'].apply(clean_date)
            
            # Print date statistics
            print("\nDate Statistics:")
            print(f"Total rows: {len(df)}")
            print(f"Valid dates: {df['Registration_date'].notna().sum()}")
            print(f"Missing dates: {df['Registration_date'].isna().sum()}")
            print("\nDate range:")
            print(f"Earliest: {df['Registration_date'].min()}")
            print(f"Latest: {df['Registration_date'].max()}")
            
            # Define important columns
            main_columns = [
                'Ditta', 'Forma giuridica', 'Sede', 'IDI', 'Registration_date',
                'Registered_address', 'Purpose', 'Address', 'Phone Numbers',
                'Email', 'Website', 'Category', 'Subcategory'
            ]
            
            # Split dataframe into main and metadata
            main_df = df[main_columns].copy()
            metadata_df = df.drop(columns=main_columns).copy()
            metadata_df['IDI'] = df['IDI']  # Keep IDI as foreign key
            
            # Create connection
            conn = sqlite3.connect(db_path)
            
            # Clean column names
            main_df.columns = [col.lower().replace(' ', '_') for col in main_df.columns]
            metadata_df.columns = [col.lower().replace(' ', '_') for col in metadata_df.columns]
            
            # Ensure registration_date is stored as DATE in SQLite
            main_df['registration_date'] = main_df['registration_date'].dt.strftime('%Y-%m-%d')
            
            # Create tables with proper date type
            main_df.to_sql('companies', conn, if_exists='replace', index=False,
                          dtype={'registration_date': 'DATE'})
            metadata_df.to_sql('company_metadata', conn, if_exists='replace', index=False)
            
            # Create indices
            with conn:
                # Indices for companies table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_idi ON companies(idi)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON companies(category)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_subcategory ON companies(subcategory)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_ditta ON companies(ditta)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sede ON companies(sede)')
                
                # Index for metadata table
                conn.execute('CREATE INDEX IF NOT EXISTS idx_metadata_idi ON company_metadata(idi)')
                
                # Create view for companies with contact info
                conn.execute('''
                    CREATE VIEW IF NOT EXISTS v_companies_with_contacts AS
                    SELECT *
                    FROM companies
                    WHERE phone_numbers IS NOT NULL 
                       OR email IS NOT NULL 
                       OR website IS NOT NULL
                ''')
                
                # Create view for category statistics
                conn.execute('''
                    CREATE VIEW IF NOT EXISTS v_category_stats AS
                    SELECT 
                        category,
                        COUNT(*) as total,
                        SUM(CASE WHEN phone_numbers IS NOT NULL THEN 1 ELSE 0 END) as has_phone,
                        SUM(CASE WHEN email IS NOT NULL THEN 1 ELSE 0 END) as has_email,
                        SUM(CASE WHEN website IS NOT NULL THEN 1 ELSE 0 END) as has_website
                    FROM companies
                    GROUP BY category
                ''')

            # Print statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM companies")
            total_companies = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM company_metadata")
            total_metadata = cursor.fetchone()[0]
            
            self.logger.info(f"\nDatabase created at {db_path}")
            self.logger.info(f"Companies table: {total_companies} records")
            self.logger.info(f"Metadata table: {total_metadata} records")
            
            # Print category distribution
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM companies
                GROUP BY category
                ORDER BY count DESC
            """)
            self.logger.info("\nCategory distribution:")
            for category, count in cursor.fetchall():
                percentage = (count / total_companies) * 100
                self.logger.info(f"{category}: {count} ({percentage:.1f}%)")

        except Exception as e:
            self.logger.error(f"Error creating database: {str(e)}")
            raise
        
        finally:
            if 'conn' in locals():
                conn.close()

def remove_duplicates_from_csv(input_file='scraped_query_results.csv', output_file=None):
    """
    Removes duplicate entries from the scraped results CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the deduplicated CSV. 
                                   If None, overwrites the input file.
    
    Returns:
        tuple: (num_total, num_unique, num_duplicates) - Statistics about the deduplication
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Store initial count
        initial_count = len(df)
        
        # Remove duplicates based on IDI (assuming this is the unique identifier)
        df_unique = df.drop_duplicates(subset=['IDI'], keep='first')
        
        # Get final count
        final_count = len(df_unique)
        duplicates_removed = initial_count - final_count
        
        # Save the deduplicated data
        if output_file is None:
            output_file = input_file
        df_unique.to_csv(output_file, index=False)
        
        print(f"Deduplication complete:")
        print(f"Total entries: {initial_count}")
        print(f"Unique entries: {final_count}")
        print(f"Duplicates removed: {duplicates_removed}")
        
        return initial_count, final_count, duplicates_removed
        
    except Exception as e:
        print(f"Error during deduplication: {str(e)}")
        return None

def clean_estratto_results(input_file='estratto_results.csv', output_file=None):
    """
    Removes specified columns from the estratto results CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the cleaned CSV.
                                   If None, overwrites the input file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Store initial columns
        initial_columns = list(df.columns)
        
        # Columns to remove
        columns_to_remove = ['idi', 'company_name']
        
        # Remove the specified columns
        df = df.drop(columns=columns_to_remove, errors='ignore')
        
        # Save the cleaned data
        if output_file is None:
            output_file = input_file
        df.to_csv(output_file, index=False)
        
        # Get final columns
        final_columns = list(df.columns)
        
        print(f"\nCleaning complete:")
        print(f"Initial columns: {', '.join(initial_columns)}")
        print(f"Removed columns: {', '.join(columns_to_remove)}")
        print(f"Remaining columns: {', '.join(final_columns)}")
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error cleaning CSV: {str(e)}")

def create_database_schema(conn):
    """
    Creates the database schema with proper tables and relationships.
    """
    cursor = conn.cursor()
    
    # Create companies table (main table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            IDI TEXT PRIMARY KEY,
            Ditta TEXT,
            Forma giuridica TEXT,
            Sede TEXT,
            Registration_date TEXT,
            Registered_address TEXT,
            Purpose TEXT
        )
    """)
    
    # Create categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Category TEXT UNIQUE,
            description TEXT
        )
    """)
    
    # Create subcategories table with foreign key to categories
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subcategories (
            subcategory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER,
            Subcategory TEXT,
            FOREIGN KEY (category_id) REFERENCES categories(category_id),
            UNIQUE(category_id, Subcategory)
        )
    """)
    
    # Create company_categories table (junction table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_categories (
            IDI TEXT,
            category_id INTEGER,
            subcategory_id INTEGER,
            FOREIGN KEY (IDI) REFERENCES companies(IDI),
            FOREIGN KEY (category_id) REFERENCES categories(category_id),
            FOREIGN KEY (subcategory_id) REFERENCES subcategories(subcategory_id),
            PRIMARY KEY (IDI)
        )
    """)
    
    conn.commit()

def csv_to_sqlite(csv_file='estratto_results.csv', 
                 db_file='zefix_data.db'):
    """
    Transforms CSV data into multiple SQLite tables with proper relationships.
    
    Args:
        csv_file (str): Path to the input CSV file
        db_file (str): Path to the SQLite database file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Ensure IDI is unique
        if df['IDI'].duplicated().any():
            print("Warning: Duplicate IDIs found. Keeping only the first occurrence.")
            df = df.drop_duplicates(subset=['IDI'], keep='first')
        
        # Create database connection
        conn = sqlite3.connect(db_file)
        
        # Create schema
        create_database_schema(conn)
        
        # Prepare data for different tables
        companies_data = df[[
            'IDI', 'Ditta', 'Forma giuridica', 'Sede', 
            'Registration_date', 'Registered_address', 'Purpose'
        ]].copy()
        
        # Get unique categories and subcategories
        categories_data = df[['Category']].drop_duplicates()
        subcategories_data = df[['Category', 'Subcategory']].drop_duplicates()
        
        # Insert data into tables
        cursor = conn.cursor()
        
        # Insert companies
        companies_data.to_sql('companies', conn, if_exists='replace', index=False)
        
        # Insert categories
        for _, row in categories_data.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO categories (Category)
                VALUES (?)
            """, (row['Category'],))
        
        # Insert subcategories
        for _, row in subcategories_data.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO categories (Category)
                VALUES (?)
            """, (row['Category'],))
            
            cursor.execute("""
                SELECT category_id FROM categories WHERE Category = ?
            """, (row['Category'],))
            category_id = cursor.fetchone()[0]
            
            cursor.execute("""
                INSERT OR IGNORE INTO subcategories (category_id, Subcategory)
                VALUES (?, ?)
            """, (category_id, row['Subcategory']))
        
        # Insert company_categories relationships
        for _, row in df.iterrows():
            # Get category_id
            cursor.execute("""
                SELECT category_id FROM categories WHERE Category = ?
            """, (row['Category'],))
            category_id = cursor.fetchone()[0]
            
            # Get subcategory_id
            cursor.execute("""
                SELECT subcategory_id 
                FROM subcategories 
                WHERE category_id = ? AND Subcategory = ?
            """, (category_id, row['Subcategory']))
            subcategory_id = cursor.fetchone()[0]
            
            # Insert relationship
            cursor.execute("""
                INSERT OR REPLACE INTO company_categories (IDI, category_id, subcategory_id)
                VALUES (?, ?, ?)
            """, (row['IDI'], category_id, subcategory_id))
        
        # Create indexes
        print("\nCreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_companies_ditta ON companies(Ditta);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_category ON categories(Category);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subcategories_subcategory ON subcategories(Subcategory);")
        
        # Print database statistics
        print("\nDatabase Statistics:")
        print("-------------------")
        cursor.execute("SELECT COUNT(*) FROM companies;")
        print(f"Companies: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM categories;")
        print(f"Categories: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM subcategories;")
        print(f"Subcategories: {cursor.fetchone()[0]}")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"\nDatabase saved to {db_file}")
        
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        if 'conn' in locals():
            conn.close()

def main():
    """Execute the database creation"""
    db_manager = DatabaseManager()
    db_manager.create_database(
        classified_file='classified_companies_openai_full.csv',
        db_path='classified_companies.db'
    )

if __name__ == "__main__":
    main()
