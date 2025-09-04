import pandas as pd
import re
import csv
from typing import List, Tuple, Dict
import string

class NERDatasetConverter:
    def __init__(self):
        """
        Initialize the converter with entity column mappings.
        These correspond to your CSV columns.
        """
        self.entity_columns = [
            'house_number', 'plot_number', 'floor','road_details', 'khasra_number', 
            'block', 'apartment', 'landmark', 'locality', 'area', 'locality2','village',
            'pincode' 
        ]
        
    def clean_and_tokenize(self, text: str) -> List[str]:
        """
        Clean and tokenize the address text.
        Updated to enforce clean splitting around pincodes and symbols.
        """
        if pd.isna(text) or text == '':
            return []

        text = str(text).strip()

        # Space-pad 6-digit numeric values for easy token recognition
        text = re.sub(r'(\D)(\d{6})(?!\d)', r'\1 \2 ', text)
        text = re.sub(r'(\d{6})(\D)', r' \1 \2', text)

        # Tokenize keeping punctuation and 6-digit numbers intact
        tokens = re.findall(r'\b\d{6}\b|\w+|[^\w\s]', text)
        return [t.strip() for t in tokens if t.strip()]
        
    def find_entity_spans(self, tokens: List[str], entity_value: str) -> List[Tuple[int, int]]:
        if pd.isna(entity_value) or entity_value == '':
            return []

        entity_tokens = self.clean_and_tokenize(str(entity_value))
        if not entity_tokens:
            return []

        spans = []
        entity_len = len(entity_tokens)

        # 🔍 Try normal span match
        for i in range(len(tokens) - entity_len + 1):
            if all(tokens[i + j].lower() == entity_tokens[j].lower() for j in range(entity_len)):
                spans.append((i, i + entity_len - 1))

        # 🛟 Fallback for PINCODE: exact 6-digit match regardless of surroundings
        if not spans and len(entity_tokens) == 1 and re.fullmatch(r'\d{6}', entity_tokens[0]):
            for i, token in enumerate(tokens):
                if token == entity_tokens[0]:
                    # Check context — ignore if preceded by "-"?
                    if i > 0 and tokens[i - 1] == '-':
                        spans.append((i, i))
                    elif i == 0 or tokens[i - 1] != '-':
                        spans.append((i, i))
                    break

        return spans

    def create_bio_tags(self, tokens: List[str], row: pd.Series) -> List[str]:
        #Create BIO tags for the tokens based on entity values in the row.
        #Uses proper BIO tagging scheme:
        #- B-ENTITY: Beginning of entity
        #- I-ENTITY: Inside/continuation of entity  
        #- O: Outside any entity
        tags = ['O'] * len(tokens)
        for column in self.entity_columns:
            if column in row and not pd.isna(row[column]) and row[column] != '':
                entity_value = str(row[column]).strip()
                spans = self.find_entity_spans(tokens, entity_value)

                for start_idx, end_idx in spans:
                    if all(tags[i] == 'O' for i in range(start_idx, end_idx + 1)):
                        tags[start_idx] = f'B-{column.upper()}'
                        for i in range(start_idx + 1, end_idx + 1):
                            tags[i] = f'I-{column.upper()}'

        return tags

    
    def convert_row(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """
        Convert a single row to tokens and BIO tags.
        """
        complete_address = row.get('complete_address', '')
        if pd.isna(complete_address) or complete_address == '':
            return [], []
        
        tokens = self.clean_and_tokenize(complete_address)
        if not tokens:
            return [], []
        
        tags = self.create_bio_tags(tokens, row)
        return tokens, tags
    
    #
    
    def convert_dataset(self, input_csv_path: str, output_csv_path: str):
        """
        Convert the entire dataset from column format to sentence-tags format.
        """
        print(f"Reading dataset from {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        df["pincode"] = df["pincode"].apply(lambda x: str(int(float(x))).zfill(6) if pd.notna(x) and str(x).strip() != '' else '')
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        print("after lower casing",df.head(5))
        print(f"Found {len(df)} rows")
        
        converted_data = []
        skipped_rows = 0
        
        for idx, row in df.iterrows():
            tokens, tags = self.convert_row(row)
            
            if tokens and tags and len(tokens) == len(tags):
                # Join tokens and tags with spaces
                sentence = ' '.join(tokens)
                tags_str = ' '.join(tags)
                converted_data.append({
                    'sentence': sentence,
                    'tags': tags_str
                })
            else:
                skipped_rows += 1
                print(f"Skipped row {idx}: Invalid tokenization")
        
        print(f"Successfully converted {len(converted_data)} rows")
        print(f"Skipped {skipped_rows} rows due to processing errors")
        
        # Save to CSV
        print(f"Saving to {output_csv_path}...")
        output_df = pd.DataFrame(converted_data)
        output_df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
        
        print("Conversion completed successfully!")
        
        # Count pincode tags in output
        pincode_tag_count = 0
        for data in converted_data:
            tags = data['tags'].split()
            pincode_tag_count += sum(1 for tag in tags if 'PINCODE' in tag)
        
        print(f"Total PINCODE tags generated: {pincode_tag_count}")
        
        # Print sample for verification
        if len(converted_data) > 0:
            print(f"\nSample BIO-tagged data:")
            for i in range(min(3, len(converted_data))):
                print(f"\nExample {i+1}:")
                print(f"Sentence: {converted_data[i]['sentence']}")
                
                # Show BIO alignment clearly
                tokens = converted_data[i]['sentence'].split()
                bio_tags = converted_data[i]['tags'].split()
                
                print("Token -> BIO Tag:")
                for token, bio_tag in zip(tokens, bio_tags):
                    marker = " ⭐" if 'PINCODE' in bio_tag else ""
                    print(f"  {token:15} -> {bio_tag}{marker}")
                print("-" * 40)
    
    def validate_conversion(self, csv_path: str, num_samples: int = 5):
        """
        Validate the converted BIO-tagged dataset by showing sample alignments.
        """
        print(f"Validating BIO-tagged dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset contains {len(df)} samples")
        print(f"Showing first {min(num_samples, len(df))} samples:\n")
        
        # Count BIO tag distribution
        all_tags = []
        for idx in range(len(df)):
            tags = df.iloc[idx]['tags'].split()
            all_tags.extend(tags)
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        print("BIO Tag Distribution:")
        for tag, count in sorted(tag_counts.items()):
            marker = " ⭐" if 'PINCODE' in tag else ""
            print(f"  {tag}: {count}{marker}")
        print()
        
        for idx in range(min(num_samples, len(df))):
            row = df.iloc[idx]
            tokens = row['sentence'].split()
            bio_tags = row['tags'].split()
            
            print(f"Sample {idx + 1}:")
            print("Token -> BIO Tag alignment:")
            for token, bio_tag in zip(tokens, bio_tags):
                marker = " ⭐" if 'PINCODE' in bio_tag else ""
                print(f"  {token:15} -> {bio_tag}{marker}")
            print("-" * 40)

# Usage example and main execution
if __name__ == "__main__":
    converter = NERDatasetConverter()
    
    input_file = "nlp_train_data(54721, 14)(141435,191023).csv"
    output_file = "bio_tagged_data_141435_191023_train.csv"
    
    try:
        # Skip pincode debug
        print("\nProceeding with conversion...")
        converter.convert_dataset(input_file, output_file)

        print("\n" + "=" * 50)
        converter.validate_conversion(output_file)

    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please update the input_file variable with the correct path to your CSV file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        
    print("\nConversion process completed!")
    print(f"Your BIO-tagged NER training data is ready in: {output_file}")
