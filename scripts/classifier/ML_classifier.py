import pandas as pd
import time
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
from transformers import pipeline
import torch
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

# Load categories from JSON file
with open('classifications.json', 'r', encoding='utf-8') as f:
    CLASSIFICATIONS = json.load(f)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')  # Open Multilingual Wordnet
    print("NLTK data downloaded successfully")

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with basic tools"""
        # Basic stopwords in multiple languages
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', # English
            'il', 'la', 'le', 'i', 'gli', 'e', 'o', 'ma', 'in', 'su', 'a',
            'per', 'di', 'con', 'da', # Italian
            'der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'auf', 'bei',
            'für', 'von', 'mit', # German
            'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'sur', 'à',
            'pour', 'de', 'avec', 'par' # French
        }
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning without NLTK"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Split into words and remove stopwords
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        
        # Remove extra whitespace and join
        return ' '.join(words)
    
    def process(self, text: str) -> str:
        """Apply text preprocessing"""
        return self.clean_text(text)

class CompanyClassifier:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        """
        Initialize the classifier with a zero-shot classification model.
        """
        try:
            # Set up GPU if available
            self.device = 0 if torch.cuda.is_available() else -1
            print(f"Using {'GPU' if self.device == 0 else 'CPU'} for inference")
            
            # Initialize the classifier
            print(f"Loading model {model_name}...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self.device,
                batch_size=4
            )
            print("Model loaded successfully")
            
            self.preprocessor = TextPreprocessor()
            print("Text preprocessor initialized")
            
            # Prepare categories and subcategories
            self.categories = [cat['category'] for cat in CLASSIFICATIONS['categories']]
            self.category_subcategories = {
                cat['category']: cat['subcategories'] 
                for cat in CLASSIFICATIONS['categories']
            }
            print(f"Loaded {len(self.categories)} categories")
            
        except Exception as e:
            print(f"Error initializing classifier: {str(e)}")
            raise

    def classify_batch(self, texts: List[str], labels: List[str], 
                      hypothesis_template: str) -> List[Dict]:
        """
        Classify a batch of texts at once.
        """
        return self.classifier(
            texts,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=8
        )

    def classify_companies_batch(self, companies: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Classify a batch of companies at once with preprocessed text.
        """
        try:
            # Debug print
            print("\nProcessing batch of", len(companies), "companies")
            
            # Preprocess all texts in the batch
            processed_texts = []
            for name, purpose in companies:
                processed_text = self.preprocessor.process(f"Company: {name}. Purpose: {purpose}")
                processed_texts.append(processed_text)
                print(f"\nProcessed text: {processed_text}")  # Debug print
            
            # Classify main categories
            print("\nClassifying categories...")  # Debug print
            category_results = self.classify_batch(
                processed_texts,
                self.categories,
                "This is a {} company."
            )
            print("\nCategory results:", category_results)  # Debug print
            
            results = []
            for idx, result in enumerate(category_results):
                print(f"\nProcessing result {idx + 1}/{len(category_results)}")  # Debug print
                category = result['labels'][0]
                category_score = result['scores'][0]
                print(f"Category: {category}, Score: {category_score}")  # Debug print
                
                if category_score < 0.5:
                    print("Low confidence, assigning to Other")  # Debug print
                    results.append(("Other", "Other"))
                    continue
                
                # Classify subcategory
                print(f"Classifying subcategory for {category}")  # Debug print
                subcategories = self.category_subcategories[category]
                subcategory_result = self.classifier(
                    processed_texts[idx],
                    candidate_labels=subcategories,
                    hypothesis_template="This company is a {}.",
                    multi_label=False
                )
                
                subcategory = subcategory_result['labels'][0]
                subcategory_score = subcategory_result['scores'][0]
                print(f"Subcategory: {subcategory}, Score: {subcategory_score}")  # Debug print
                
                if subcategory_score < 0.5:
                    print("Low subcategory confidence, using category as subcategory")  # Debug print
                    subcategory = category
                
                results.append((category, subcategory))
                
            return results

        except Exception as e:
            print(f"Error in classify_companies_batch: {str(e)}")
            print(f"Full error: {e.__class__.__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return [("Other", "Other")] * len(companies)

    def process_csv(self, input_file: str = 'estratto_results.csv', 
                   output_file: str = 'classified_companies_ml.csv',
                   batch_size: int = 4,  # Reduced for testing
                   limit: int = None) -> None:
        """
        Process the CSV file and add classifications.
        Uses batched processing for speed.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            
            if limit is not None:
                df = df.head(limit)
                print(f"\nProcessing first {limit} companies...")
            
            # Add classification columns
            if 'Category' not in df.columns:
                df['Category'] = None
            if 'Subcategory' not in df.columns:
                df['Subcategory'] = None
            
            # Load existing progress
            existing_df = self._load_progress(output_file)
            if existing_df is not None:
                print("\nFound existing progress. Resuming from last saved state...")
                for col in ['Category', 'Subcategory']:
                    if col in existing_df.columns:
                        df[col] = existing_df[col]
            
            total_rows = len(df)
            processed_count = len(df[df['Category'].notna()])
            
            print(f"\nTotal companies: {total_rows}")
            print(f"Already processed: {processed_count}")
            print(f"Remaining: {total_rows - processed_count}")
            
            # Process in larger batches
            unprocessed_mask = df['Category'].isna() & df['Purpose'].notna()
            unprocessed_indices = df[unprocessed_mask].index
            
            for i in tqdm(range(0, len(unprocessed_indices), batch_size), 
                         desc="Processing batches"):
                batch_indices = unprocessed_indices[i:i + batch_size]
                batch_data = [(df.at[idx, 'Ditta'], df.at[idx, 'Purpose']) 
                             for idx in batch_indices]
                
                if not batch_data:
                    continue
                
                try:
                    # Process batch
                    results = self.classify_companies_batch(batch_data)
                    
                    # Update dataframe
                    for idx, (category, subcategory) in zip(batch_indices, results):
                        df.at[idx, 'Category'] = category
                        df.at[idx, 'Subcategory'] = subcategory
                    
                    # Save progress
                    df.to_csv(output_file, index=False)
                    processed_count += len(batch_data)
                    print(f"\nProgress: {processed_count}/{total_rows}")
                    
                except Exception as e:
                    print(f"\nError processing batch: {str(e)}")
                    continue
            
            print(f"\nClassification completed. Results saved to {output_file}")
            self._print_statistics(df, total_rows)
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            if 'df' in locals():
                error_output = 'error_' + output_file
                df.to_csv(error_output, index=False)
                print(f"Progress saved to {error_output}")

    def _load_progress(self, output_file: str) -> pd.DataFrame:
        """
        Load existing progress from output file if it exists.
        """
        if os.path.exists(output_file):
            return pd.read_csv(output_file)
        return None

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
        # Initialize classifier
        classifier = CompanyClassifier()
        
        # Ask for limit
        use_limit = input("Do you want to set a limit? (y/n): ").lower()
        limit = int(input("Enter limit: ")) if use_limit == 'y' else None
        
        # Run classification
        classifier.process_csv(limit=limit)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
