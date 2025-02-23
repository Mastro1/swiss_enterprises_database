import pandas as pd
import time
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
from dotenv import load_dotenv
import json
import google.generativeai as genai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Load categories from JSON file
with open('classifications.json', 'r', encoding='utf-8') as f:
    CLASSIFICATIONS = json.load(f)

# Gemini Pro pricing (as of 2024)
INPUT_PRICE_PER_1K = 0.000075  # $0.000075 per 1K input tokens
OUTPUT_PRICE_PER_1K = 0.0005   # $0.0005 per 1K output tokens

# Gemini model configuration
GEMINI_CONFIG = {
    'model': 'gemini-2.0-flash-lite-preview-02-05',
    'generation_config': {
        'temperature': 0.2,  # Lower temperature for more consistent outputs
        'top_p': 0.8,       # Nucleus sampling
        'top_k': 40,        # Top-k sampling
        'max_output_tokens': 100,  # Limit output size since we need just the category
        'candidate_count': 1,      # We only need one response
    },
    'safety_settings': [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
}

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    Uses the same encoding as GPT-3.5-turbo and GPT-4.
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0

def estimate_total_tokens(df: pd.DataFrame) -> Dict[str, int]:
    """
    Estimate the total number of tokens that will be used for classification.
    Returns a dictionary with token counts for different components.
    """
    # Create sample prompt to calculate base prompt tokens
    categories_text = ""
    for cat in CLASSIFICATIONS['categories']:
        categories_text += f"\n{cat['category']}:\n"
        categories_text += ", ".join(cat['subcategories'])
        categories_text += "\n"
    
    base_prompt = f"""
    Classify the following company into the most appropriate category and subcategory.
    
    Company Name: "Sample Company"
    Company Purpose: "Sample Purpose"

    Available categories and subcategories:
    {categories_text}

    Consider both the name and purpose of the company. The text might be in Italian, German, or French, 
    but classify it into the English categories provided.

    Important guidelines:
    1. Focus on the main business activity
    2. Consider both name and purpose equally
    3. If multiple categories could apply, choose the most specific one
    4. Return ONLY the category and subcategory in this exact format:
       Category: [category name]
       Subcategory: [subcategory name]
    """
    
    base_tokens = count_tokens(base_prompt)
    
    # Calculate tokens for all company names and purposes
    company_tokens = sum(count_tokens(str(name)) for name in df['Ditta'])
    purpose_tokens = sum(count_tokens(str(purpose)) for purpose in df['Purpose'])
    
    # Estimate response tokens (usually much smaller than input)
    estimated_response_tokens = len(df) * 20  # Approximate tokens per response
    
    total_tokens = base_tokens * len(df) + company_tokens + purpose_tokens + estimated_response_tokens
    
    return {
        'base_prompt_tokens': base_tokens,
        'company_name_tokens': company_tokens,
        'purpose_tokens': purpose_tokens,
        'estimated_response_tokens': estimated_response_tokens,
        'total_estimated_tokens': total_tokens,
        'number_of_companies': len(df),
        'average_tokens_per_company': total_tokens / len(df) if len(df) > 0 else 0
    }

def print_token_estimate(csv_file: str = 'estratto_results.csv') -> None:
    """
    Print a detailed token usage estimate and approximate cost.
    """
    try:
        df = pd.read_csv(csv_file)
        token_stats = estimate_total_tokens(df)
        
        print("\nToken Usage Estimate:")
        print("--------------------")
        print(f"Number of companies to process: {token_stats['number_of_companies']}")
        print(f"Base prompt tokens (per company): {token_stats['base_prompt_tokens']}")
        print(f"Total company name tokens: {token_stats['company_name_tokens']}")
        print(f"Total purpose tokens: {token_stats['purpose_tokens']}")
        print(f"Estimated response tokens: {token_stats['estimated_response_tokens']}")
        print(f"Average tokens per company: {token_stats['average_tokens_per_company']:.1f}")
        print(f"\nTotal estimated tokens: {token_stats['total_estimated_tokens']}")
        
        input_tokens = token_stats['base_prompt_tokens'] * token_stats['number_of_companies'] + \
                      token_stats['company_name_tokens'] + token_stats['purpose_tokens']
        output_tokens = token_stats['estimated_response_tokens']
        
        estimated_input_cost = (input_tokens / 1000) * INPUT_PRICE_PER_1K
        estimated_output_cost = (output_tokens / 1000) * OUTPUT_PRICE_PER_1K
        total_cost = estimated_input_cost + estimated_output_cost
        
        print("\nEstimated Costs (USD):")
        print("---------------------")
        print(f"Input cost: ${estimated_input_cost:.2f}")
        print(f"Output cost: ${estimated_output_cost:.2f}")
        print(f"Total estimated cost: ${total_cost:.2f}")
        
    except Exception as e:
        print(f"Error estimating tokens: {str(e)}")

class CompanyClassifier:
    def __init__(self, api_key: str = None):
        """
        Initialize the classifier with Google API key.
        If api_key is None, tries to get it from environment variable GOOGLE_API_KEY.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_CONFIG['model'],
            generation_config=GEMINI_CONFIG['generation_config'],
            safety_settings=GEMINI_CONFIG['safety_settings']
        )
        self.requests_this_minute = 0
        self.last_request_time = time.time()
        self.rate_limit = 10  # Changed to 10 requests per minute

    def _check_rate_limit(self):
        """
        Check and handle rate limiting.
        Sleeps if necessary to stay within limits.
        """
        current_time = time.time()
        if current_time - self.last_request_time >= 60:
            # Reset counter after a minute
            self.requests_this_minute = 0
            self.last_request_time = current_time
        elif self.requests_this_minute >= self.rate_limit:
            # Wait until the next minute starts
            sleep_time = 60 - (current_time - self.last_request_time)
            print(f"\nRate limit reached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            self.requests_this_minute = 0
            self.last_request_time = time.time()

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3)
    )
    def classify_company(self, name: str, purpose: str) -> Tuple[str, str]:
        """
        Classify a company based on both its name and purpose.
        Returns both category and subcategory.
        """
        try:
            # Check rate limit before making request
            self._check_rate_limit()
            
            # Create a formatted string of categories and subcategories
            categories_text = ""
            for cat in CLASSIFICATIONS['categories']:
                categories_text += f"\n{cat['category']}:\n"
                categories_text += ", ".join(cat['subcategories'])
                categories_text += "\n"

            prompt = f"""
            Classify the following company into the most appropriate category and subcategory.
            
            Company Name: "{name}"
            Company Purpose: "{purpose}"

            Available categories and subcategories:
            {categories_text}

            Consider both the name and purpose of the company. The text might be in Italian, German, or French, 
            but classify it into the English categories provided.

            Important guidelines:
            1. Focus on the main business activity
            2. If multiple categories could apply, choose the most specific one
            3. Return ONLY the category and subcategory in this exact format:
               Category: [category name]
               Subcategory: [subcategory name]

            If no subcategory fits well, use the category name as the subcategory.
            """

            response = self.model.generate_content(
                prompt,
                stream=False
            )
            self.requests_this_minute += 1
            response_text = response.text.strip()
            
            # Parse the response to extract category and subcategory
            lines = response_text.split('\n')
            print("Response text:", response_text)  # Debug line
            
            try:
                category = lines[0].replace('Category:', '').strip()
                subcategory = lines[1].replace('Subcategory:', '').strip()
                
                # Clean up any colons in category names
                if ':' in category:
                    category = category.split(':')[0].strip()
                
                print("Category:", category)
                print("Subcategory:", subcategory)
                
                # Validate category
                valid_categories = [cat['category'] for cat in CLASSIFICATIONS['categories']]
                if category not in valid_categories:
                    print(f"Invalid category: {category}")
                    return "Other", "Other"
                
                # Validate subcategory
                category_data = next((cat for cat in CLASSIFICATIONS['categories'] if cat['category'] == category), None)
                if subcategory not in category_data['subcategories']:
                    print(f"Invalid subcategory: {subcategory}")
                    subcategory = category
                
                return category, subcategory
                
            except IndexError:
                print("Error parsing response:", response_text)
                return "Other", "Other"

        except Exception as e:
            if "429" in str(e):
                print(f"\nRate limit exceeded. Retrying after backoff...")
                raise  # This will trigger the retry mechanism
            else:
                print(f"Error classifying company: {str(e)}")
                return "Other", "Other"

    def _load_progress(self, output_file: str) -> pd.DataFrame:
        """
        Load existing progress from output file if it exists.
        """
        if os.path.exists(output_file):
            return pd.read_csv(output_file)
        return None

    def process_csv(self, input_file: str = 'contact_df.csv', 
                   output_file: str = 'classified_companies_gemini.csv',
                   batch_size: int = 5,  # Reduced batch size due to rate limit
                   limit: int = None) -> None:
        """
        Process the CSV file and add classifications.
        Includes progress saving and rate limiting.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            
            # Apply limit if specified
            if limit is not None:
                df = df.head(limit)
                print(f"\nProcessing first {limit} companies...")
            
            # Load existing progress
            existing_df = self._load_progress(output_file)
            if existing_df is not None:
                print("\nFound existing progress. Resuming from last saved state...")
                # Update only unclassified entries
                for col in ['Category', 'Subcategory']:
                    if col in existing_df.columns:
                        df[col] = existing_df[col]
            
            total_rows = len(df)
            processed_count = len(df[df['Category'].notna()])
            
            print(f"\nTotal companies: {total_rows}")
            print(f"Already processed: {processed_count}")
            print(f"Remaining: {total_rows - processed_count}")
            
            # Add classification columns if they don't exist
            if 'Category' not in df.columns:
                df['Category'] = None
            if 'Subcategory' not in df.columns:
                df['Subcategory'] = None
            
            # Process in batches
            batch_count = 0
            for i in tqdm(range(0, total_rows, batch_size), desc="Processing companies"):
                batch = df.iloc[i:i+batch_size]
                batch_modified = False
                
                # Only process rows without categories
                for idx, row in batch.iterrows():
                    if pd.isna(df.at[idx, 'Category']) and not pd.isna(row['Purpose']):
                        try:
                            category, subcategory = self.classify_company(
                                name=row['Ditta'],
                                purpose=row['Purpose']
                            )
                            df.at[idx, 'Category'] = category
                            df.at[idx, 'Subcategory'] = subcategory
                            batch_modified = True
                            
                            # Print progress
                            processed_count += 1
                            print(f"\nProgress: {processed_count}/{total_rows} companies processed")
                            
                        except Exception as e:
                            print(f"\nError processing company {row['Ditta']}: {str(e)}")
                            continue
                
                # Save after each modified batch
                if batch_modified:
                    df.to_csv(output_file, index=False)
                    batch_count += 1
                    print(f"Saved batch {batch_count} to {output_file}")
                
                # Reduced sleep time but still maintain some gap
                time.sleep(0.1)
            
            print(f"\nClassification completed. Results saved to {output_file}")
            
            # Print statistics
            self._print_statistics(df, total_rows)
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            # Save progress even if there's an error
            if 'df' in locals():
                error_output = 'error_' + output_file
                df.to_csv(error_output, index=False)
                print(f"Progress saved to {error_output}")

    def _print_statistics(self, df: pd.DataFrame, total_rows: int):
        """
        Print detailed statistics about the classification results.
        """
        print("\nCategory Distribution:")
        category_counts = df['Category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / total_rows) * 100
            print(f"{category}: {count} companies ({percentage:.1f}%)")
            
            # Print subcategory distribution for this category
            subcategory_counts = df[df['Category'] == category]['Subcategory'].value_counts()
            for subcategory, subcount in subcategory_counts.items():
                sub_percentage = (subcount / count) * 100
                print(f"  - {subcategory}: {subcount} ({sub_percentage:.1f}%)")
        
        # Print number of unclassified companies
        unclassified = df['Category'].isna().sum()
        if unclassified > 0:
            print(f"\nUnclassified companies: {unclassified}")

def main():
    """
    Main function to run the classification process.
    """
    try:
        # Set test limit
        TEST_LIMIT = 20
        
        # First show token estimate
        print(f"Calculating token usage estimate for first {TEST_LIMIT} companies...")
        df = pd.read_csv('estratto_results.csv').head(TEST_LIMIT)
        token_stats = estimate_total_tokens(df)
        
        print("\nToken Usage Estimate:")
        print("--------------------")
        print(f"Number of companies to process: {token_stats['number_of_companies']}")
        print(f"Base prompt tokens (per company): {token_stats['base_prompt_tokens']}")
        print(f"Total company name tokens: {token_stats['company_name_tokens']}")
        print(f"Total purpose tokens: {token_stats['purpose_tokens']}")
        print(f"Estimated response tokens: {token_stats['estimated_response_tokens']}")
        print(f"Average tokens per company: {token_stats['average_tokens_per_company']:.1f}")
        print(f"\nTotal estimated tokens: {token_stats['total_estimated_tokens']}")
        
        # Calculate costs
        input_tokens = token_stats['base_prompt_tokens'] * token_stats['number_of_companies'] + \
                      token_stats['company_name_tokens'] + token_stats['purpose_tokens']
        output_tokens = token_stats['estimated_response_tokens']
        
        estimated_input_cost = (input_tokens / 1000) * INPUT_PRICE_PER_1K
        estimated_output_cost = (output_tokens / 1000) * OUTPUT_PRICE_PER_1K
        total_cost = estimated_input_cost + estimated_output_cost
        
        print("\nEstimated Costs (USD):")
        print("---------------------")
        print(f"Input cost: ${estimated_input_cost:.4f}")
        print(f"Output cost: ${estimated_output_cost:.4f}")
        print(f"Total estimated cost: ${total_cost:.4f}")
        
        proceed = input("\nDo you want to proceed with classification? (y/n): ").lower()
        if proceed == 'y':
            classifier = CompanyClassifier()
            classifier.process_csv(limit=TEST_LIMIT)
        else:
            print("Classification cancelled.")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
