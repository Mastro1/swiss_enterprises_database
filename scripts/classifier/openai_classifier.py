import pandas as pd
import openai
import json
import time
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
import logging

# Load categories from JSON file
with open('classifications.json', 'r', encoding='utf-8') as f:
    CLASSIFICATIONS = json.load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='openai_classifier.log'
)

class OpenAIClassifier:
    def __init__(self, api_key: str = None):
        """Initialize the OpenAI classifier"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        self.categories = [cat['category'] for cat in CLASSIFICATIONS['categories']]
        self.category_subcategories = {
            cat['category']: cat['subcategories'] 
            for cat in CLASSIFICATIONS['categories']
        }
        
        # Price tracking
        self.total_tokens = 0
        self.PRICE_PER_1M_TOKENS = 0.15  # $0.15 per 1M tokens for GPT-4-mini

    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters)"""
        return len(text) // 4

    def calculate_cost(self) -> float:
        """Calculate cost in USD based on used tokens"""
        return (self.total_tokens / 1_000_000) * self.PRICE_PER_1M_TOKENS

    def classify_company(self, name: str, purpose: str) -> Tuple[str, str]:
        """Classify a single company using OpenAI"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Prepare the prompt for category classification
                category_prompt = f"""You are a business classifier. Given the following company information:
Name: {name}
Purpose: {purpose}

You must classify this company into exactly one of these categories:
{', '.join(self.categories)}

Rules:
1. Choose the most specific category that fits
2. Only respond with one of the listed categories
3. Do not add any explanation or additional text
4. If unsure, choose the closest matching category rather than 'Other'

Respond with only the category name."""

                self.total_tokens += self.estimate_tokens(category_prompt)

                try:
                    # Get category using new API format with timeout
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a precise business classifier that only responds with exact category names."},
                            {"role": "user", "content": category_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=50,
                        timeout=30  # 30 second timeout
                    )
                    
                    category = response.choices[0].message.content.strip()
                    print(f"Processing: {name} -> Category: {category}")  # Added progress indicator
                    
                    if category not in self.categories:
                        print(f"\nCompany: {name}")
                        print(f"Purpose: {purpose}")
                        print(f"OpenAI response: '{category}'")
                        print(f"Valid categories: {self.categories}")
                        return "Other", "Other"
                        
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return "Other", "Other"

                self.total_tokens += 50
                
                # Get subcategories for the chosen category
                subcategories = self.category_subcategories.get(category, [])
                if not subcategories:
                    return category, category

                # Prepare the prompt for subcategory classification
                subcategory_prompt = f"""You are a business classifier. Given the following company information:
Name: {name}
Purpose: {purpose}
Category: {category}

You must classify this company into exactly one of these subcategories:
{', '.join(subcategories)}

Rules:
1. Choose the most specific subcategory that fits
2. Only respond with one of the listed subcategories
3. Do not add any explanation or additional text
4. If unsure, choose the closest matching subcategory

Respond with only the subcategory name."""

                self.total_tokens += self.estimate_tokens(subcategory_prompt)

                try:
                    # Get subcategory using new API format with timeout
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a precise business classifier that only responds with exact subcategory names."},
                            {"role": "user", "content": subcategory_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=50,
                        timeout=30  # 30 second timeout
                    )
                    
                    subcategory = response.choices[0].message.content.strip()
                    
                    if subcategory not in subcategories:
                        print(f"ERROR: Invalid subcategory '{subcategory}'. Must be one of: {subcategories}")
                        return category, category
                        
                except Exception as e:
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return category, category

                self.total_tokens += 50
                return category, subcategory

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed with critical error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                logging.error(f"All attempts failed for company {name}: {str(e)}")
                return "Other", "Other"

        return "Other", "Other"  # If all retries fail

    def process_csv(self, input_file: str = 'contact_df.csv', 
                   output_file: str = 'classified_companies_openai.csv',
                   batch_size: int = 20,
                   test_mode: bool = False) -> None:
        """Process companies from CSV file with batching and progress tracking"""
        try:
            # Read input CSV
            df = pd.read_csv(input_file)
            total_rows = len(df)

            # Check for existing progress
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                print(f"\nFound existing progress file with {len(existing_df)} classified companies")
                
                # Update classifications from existing results
                classified_indices = existing_df.index[
                    (existing_df['Category'] != 'Other') | 
                    (existing_df['Subcategory'] != 'Other')
                ]
                
                # Merge existing classifications
                for idx in classified_indices:
                    df.at[idx, 'Category'] = existing_df.at[idx, 'Category']
                    df.at[idx, 'Subcategory'] = existing_df.at[idx, 'Subcategory']
                
                # Calculate remaining companies
                remaining = total_rows - len(classified_indices)
                print(f"Remaining companies to classify: {remaining}")
            else:
                # Initialize results columns
                df['Category'] = 'Other'
                df['Subcategory'] = 'Other'
                remaining = total_rows

            # Cost estimation and confirmation
            avg_tokens_per_company = 300
            estimated_total_tokens = remaining * avg_tokens_per_company
            estimated_cost = (estimated_total_tokens / 1_000_000) * self.PRICE_PER_1M_TOKENS
            
            print(f"\nEstimated cost for remaining companies: ${estimated_cost:.2f}")
            proceed = input("Do you want to proceed? (y/n): ")
            if proceed.lower() != 'y':
                print("Operation cancelled by user")
                return
            
            print(f"\nProcessing remaining {remaining} companies...")
            processed_count = total_rows - remaining
            
            # Process in batches
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                unclassified_in_batch = batch[
                    (batch['Category'] == 'Other') & 
                    (batch['Subcategory'] == 'Other')
                ]
                
                if len(unclassified_in_batch) == 0:
                    continue
                    
                with tqdm(total=len(batch), desc=f"Batch {i//batch_size + 1}") as pbar:
                    for idx, row in batch.iterrows():
                        # Skip if already classified
                        if row['Category'] != 'Other' or row['Subcategory'] != 'Other':
                            pbar.update(1)
                            continue
                            
                        try:
                            print(f"\nProcessing company {processed_count + 1}/{total_rows}")
                            category, subcategory = self.classify_company(
                                row['Ditta'],
                                row['Purpose']
                            )
                            
                            df.at[idx, 'Category'] = category
                            df.at[idx, 'Subcategory'] = subcategory
                            
                            pbar.update(1)
                            processed_count += 1
                            
                            # Save after each company
                            df.to_csv(output_file, index=False)
                            print(f"Progress saved. Current cost: ${self.calculate_cost():.2f}")
                            
                            # Reduced delay between requests
                            time.sleep(60/500)  # 500 requests per minute
                            
                        except Exception as e:
                            print(f"Error processing company: {str(e)}")
                            logging.error(f"Error processing row {idx}: {str(e)}")
                            # Save on error
                            df.to_csv(output_file, index=False)
                            time.sleep(2)  # Reduced from 5s to 2s for error delay
                            continue
                
                # Save after each batch
                df.to_csv(output_file, index=False)
                print(f"\nProgress: {processed_count}/{total_rows}")
                print(f"Current cost: ${self.calculate_cost():.2f}")
                time.sleep(1)  # Rate limiting
            
            final_cost = self.calculate_cost()
            print(f"\nClassification completed. Results saved to {output_file}")
            print(f"Final cost: ${final_cost:.2f}")
            self._print_statistics(df)
            
        except Exception as e:
            logging.error(f"Error processing CSV: {str(e)}")
            if 'df' in locals():
                df.to_csv('error_' + output_file, index=False)
            print(f"Error occurred. Current cost: ${self.calculate_cost():.2f}")

    def _print_statistics(self, df: pd.DataFrame) -> None:
        """Print classification statistics"""
        print("\nCategory Distribution:")
        category_counts = df['Category'].value_counts()
        total_rows = len(df)
        
        for category, count in category_counts.items():
            percentage = (count / total_rows) * 100
            print(f"{category}: {count} companies ({percentage:.1f}%)")
            
            # Print subcategory distribution for this category
            subcategory_counts = df[df['Category'] == category]['Subcategory'].value_counts()
            for subcategory, subcount in subcategory_counts.items():
                sub_percentage = (subcount / count) * 100
                print(f"  - {subcategory}: {subcount} ({sub_percentage:.1f}%)")

def main():
    """Main execution function"""
    classifier = OpenAIClassifier()
    
    # Run on full dataset with larger batch size for efficiency
    classifier.process_csv(
        test_mode=False,
        batch_size=20,  # Increased batch size for full run
        output_file='classified_companies_openai_full.csv'  # Distinct output file
    )

if __name__ == "__main__":
    main() 