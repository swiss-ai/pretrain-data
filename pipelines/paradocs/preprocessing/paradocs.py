import argparse
import pandas as pd
import sys
import random
import gzip
import csv
import os

# This code is based on: https://github.com/rewicks/ParaDocs/blob/main/paradocs/paradocs

csv.field_size_limit(sys.maxsize)

def is_consecutive(preceding, subsequent):
    try:
        if int(subsequent["src_start_index"]) - int(preceding["src_end_index"]) <= 2 \
                and int(subsequent["tgt_start_index"]) - int(preceding["tgt_end_index"]) <= 2:
            return True
    except (ValueError, TypeError):
        return False
    return False

def breaks_document(row, args):
    if not str(row['src']).strip() or not str(row['tgt']).strip():
        return True
    
    if float(row["src_language_id"]) < args.lid_cutoff or \
       float(row["tgt_language_id"]) < args.lid_cutoff:
        return True
        
    if int(row["frequency"]) > args.frequency_cutoff:
        return True

    return False

def process_and_concatenate_text(df, args):
    if df.empty:
        print("Input DataFrame is empty.", file=sys.stderr)
        return

    max_length = args.max_length
    def yield_bidirectional(src_text, tgt_text, src_lang, tgt_lang):
        if src_text and tgt_text:
            st = src_text.strip()
            tt = tgt_text.strip()
            yield f"{src_lang}: {st}\n{tgt_lang}: {tt}"
            yield f"{tgt_lang}: {tt}\n{src_lang}: {st}"
    
    current_src_text = ""
    current_tgt_text = ""
    last_row_in_block = None

    for index, row in df.iterrows():
        if breaks_document(row, args):
            if last_row_in_block is not None:
                yield from yield_bidirectional(current_src_text, current_tgt_text, args.src, args.tgt)
            
            current_src_text = ""
            current_tgt_text = ""
            last_row_in_block = None
            continue

        if last_row_in_block is None:
            current_src_text = str(row['src'])
            current_tgt_text = str(row['tgt'])
            last_row_in_block = row
            continue

        is_same_doc_para = (
            row['src_docid'] == last_row_in_block['src_docid'] and
            row['src_paragraph_id'] == last_row_in_block['src_paragraph_id']
        )
        is_consecutive_sentence = is_consecutive(last_row_in_block, row)
        
        would_exceed_limit = (
            (len(current_src_text) + len(str(row['src'])) +
             len(current_tgt_text) + len(str(row['tgt'])) + 2) > max_length
        )

        if is_same_doc_para and is_consecutive_sentence and not would_exceed_limit:
            current_src_text += " " + str(row['src'])
            current_tgt_text += " " + str(row['tgt'])
            last_row_in_block = row
        else:
            yield from yield_bidirectional(current_src_text, current_tgt_text, args.src, args.tgt)

            current_src_text = str(row['src'])
            current_tgt_text = str(row['tgt'])
            last_row_in_block = row

    if last_row_in_block is not None:
        yield from yield_bidirectional(current_src_text, current_tgt_text, args.src, args.tgt)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenates consecutive sentences from a gzipped TSV file into text blocks.\n"
            "Each block is generated for both src->tgt and tgt->src directions.\n"
            "The final dataset can be shuffled and sampled before being saved."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--src", type=str, required=True, help="Source language code (e.g., 'en').")
    parser.add_argument("--tgt", type=str, required=True, help="Target language code (e.g., 'de').")
    parser.add_argument("--input", type=str, nargs='?', default=None, help="Path to the input gzipped TSV file. If not provided, reads from stdin.")
    parser.add_argument("--output", type=str, default=None, help="Path to the output Parquet file. If not set, prints to standard output.")
    
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum character length of a concatenated block.")
    parser.add_argument('--frequency_cutoff', type=int, default=100, help="Break documents at any line with a frequency count above this.")
    parser.add_argument("--lid_cutoff", type=float, default=0.5, help="Break documents if language ID probability is below this.")
    
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of the final shuffled data to keep (0.0 to 1.0). Only used with --output.")

    args = parser.parse_args()
    df = None
    try:
        if args.input:
            print(f"Reading gzipped TSV from {args.input}...", file=sys.stderr)
            istream = gzip.open(args.input, 'rt', encoding='utf-8')
        else:
            print("Reading TSV from standard input...", file=sys.stderr)
            istream = sys.stdin

        reader = csv.DictReader(
            istream,
            delimiter='\t',
            quoting=csv.QUOTE_NONE,
            fieldnames=[
                "src", "tgt", "similarity_one", "similarity_two", "collection",
                "src_paragraph_id", "tgt_paragraph_id", "src_sentence_id", "tgt_sentence_id",
                "src_start_index", "src_end_index", "tgt_start_index", "tgt_end_index",
                "src_language_id", "tgt_language_id", "frequency", "src_docid", "tgt_docid"
            ]
        )
        data = list(reader)
        if not data:
            print("Error: Input stream was empty.", file=sys.stderr)
            sys.exit(1)

        df = pd.DataFrame(data)
        
        numeric_cols = [
            'src_paragraph_id', 'src_start_index', 'src_end_index',
            'tgt_start_index', 'tgt_end_index',
            'src_language_id', 'tgt_language_id', 'frequency'
        ]
        for col in numeric_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numeric_cols, inplace=True)
        if args.input:
            istream.close()
    except Exception as e:
        print(f"Error reading TSV data: {e}", file=sys.stderr)
        sys.exit(1)

    block_generator = process_and_concatenate_text(df, args)

    if args.output:
        print("Collecting generated blocks...", file=sys.stderr)
        all_blocks = list(block_generator)
        output_df = pd.DataFrame(all_blocks, columns=['text'])
        
        print(f"Generated {len(output_df)} blocks (bidirectionally).", file=sys.stderr)
        print("Shuffling data...", file=sys.stderr)
        output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)

        if args.sample_ratio < 1.0:
            num_rows_to_keep = int(len(output_df) * args.sample_ratio)
            print(f"Sampling {args.sample_ratio} of data, keeping {num_rows_to_keep} rows.", file=sys.stderr)
            output_df = output_df.head(num_rows_to_keep)

        try:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_df.to_parquet(args.output, index=False, engine='pyarrow')
            print(f"Successfully wrote {len(output_df)} blocks to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing to Parquet file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        output_stream = sys.stdout
        for i, block in enumerate(block_generator):
            if i > 0:
                output_stream.write("\n\n" + "="*40 + "\n\n")
            output_stream.write(block)
        output_stream.write("\n")
if __name__ == "__main__":
    main()