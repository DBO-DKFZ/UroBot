import camelot
import re
import fitz

import pandas as pd

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextBoxHorizontal


def aggregate_table_to_text(series, separator="; "):
    aggregated_text = separator.join(series.astype(str))

    return aggregated_text


def prepare_all_text_data(df, separator=' | ', clean=True):
    combined_text = df.apply(lambda row: separator.join(row.values.astype(str)), axis=1)

    if clean:
        # Define a cleaning function
        def clean_text(text):
            text = text.lower()  # Lowercase text
            text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
            text = re.sub(r'[\r|\n|\r\n]+', ' ', text)  # Remove line breaks
            text = re.sub(r'[\W_]+', ' ', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
            return text.strip()

        # Apply cleaning function
        cleaned_text = combined_text.apply(clean_text)
    else:
        cleaned_text = combined_text

    # Handle missing values
    cleaned_text = cleaned_text.fillna('')

    # Add the prepared text to the DataFrame
    df['prepared_text'] = cleaned_text

    df = df['prepared_text']

    return df


def extract_text_with_page_numbers(pdf_path, config):
    pages = None
    if config["filter_toc_refs"]:
        start_page, end_page = get_relevant_pages(pdf_path)
        pages = list(range(start_page, end_page + 1))

    filtered = []
    pages_per_chunk = []  # List to store lists of page numbers for each text chunk
    temp_string = ''  # Temporary string to accumulate text
    temp_pages = []  # Temporary list to track pages for the current text chunk
    current_page = 1  # Start from the first page

    for page_layout in extract_pages(pdf_path):
        if page_layout.pageid in pages or pages is None:
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text()
                    splitted = text.split("\n\n")  # Split text into paragraphs

                    for string in splitted:
                        cleaned_string = ' '.join(string.replace('\n', ' ').split())
                        if re.compile(r"\bREFERENCES\b").search(cleaned_string) and pages is not None:
                            return filtered, pages_per_chunk

                        if cleaned_string:  # Check if there is actual content after cleaning
                            if current_page not in temp_pages:
                                temp_pages.append(current_page)  # Add the current page number if not already included

                            temp_string += cleaned_string + " "  # Accumulate cleaned text

                            # Check conditions to finalize the current chunk
                            if len(temp_string) > config["chunk_threshold"] and (
                                    temp_string.endswith('. ') or temp_string.endswith('? ') or temp_string.endswith(
                                '! ')):
                                filtered.append(temp_string.strip())
                                pages_per_chunk.append(temp_pages.copy())
                                temp_string = ''  # Reset temporary string
                                temp_pages = []  # Reset page tracking for the next chunk

        current_page += 1  # Move to the next page

    # Handle any remaining text chunk after the last page
    if temp_string.strip():
        filtered.append(temp_string.strip())
        pages_per_chunk.append(temp_pages)

    return filtered, pages_per_chunk


def extract_by_char_limit(pdf_path, threshold=500):
    text = extract_text(pdf_path)
    splitted = text.split("\n\n")  # Initial split by double new lines to get paragraphs

    filtered = []
    temp_string = ''  # Temporary string to accumulate text

    for string in splitted:
        # Remove consecutive new lines within a paragraph, and trim multiple spaces
        cleaned_string = ' '.join(string.replace('\n', ' ').split())

        # Proceed with accumulation and checking against the threshold
        temp_string += cleaned_string + " "  # Add space for separation

        # Check if the accumulated string meets criteria to be added to filtered
        if len(temp_string) > threshold and (
                temp_string.endswith('. ') or temp_string.endswith('? ') or temp_string.endswith('! ')):
            filtered.append(temp_string.strip())  # Append to filtered and remove trailing space
            temp_string = ''  # Reset temporary string
        elif len(cleaned_string) > threshold:
            # If there's significant content in temp_string, add it first
            if len(temp_string.strip()) > len(cleaned_string):
                filtered.append(temp_string.strip())
                temp_string = ''  # Reset for next accumulation
            else:
                # If temp_string was mostly the current string, start fresh
                temp_string = cleaned_string + " "  # Start accumulation afresh with current string

    # Make sure to add any remaining accumulated text
    if temp_string.strip():
        filtered.append(temp_string.strip())

    return filtered


def extract_by_paragraphs(pdf_path):
    paragraphs = []
    current_paragraph = ""
    last_y0 = None

    # Set up the PDF page aggregator
    laparams = LAParams()
    resource_manager = PDFResourceManager()
    device = PDFPageAggregator(resource_manager, laparams=laparams)
    interpreter = PDFPageInterpreter(resource_manager, device)

    for page in PDFPage.get_pages(open(pdf_path, 'rb')):
        interpreter.process_page(page)
        layout = device.get_result()
        for element in layout:
            if isinstance(element, LTTextBox):
                for text_line in element:
                    # Check the y0 position to determine if this line is part of a new paragraph
                    if last_y0 is not None and (last_y0 - text_line.y0) > 100:
                        # Consider as new paragraph if the gap is big enough
                        paragraphs.append(current_paragraph.strip())
                        current_paragraph = text_line.get_text()
                    else:
                        current_paragraph += " " + text_line.get_text()
                    last_y0 = text_line.y0
    if current_paragraph.strip() != "":
        paragraphs.append(current_paragraph.strip())

    return paragraphs


def extract_tables(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

    tables_list = []
    table_pages = []

    # Iterate through tables and print them
    for i, table in enumerate(tables, start=1):
        tables_list.append(table.df)
        table_pages.append(table.page)

    return tables_list, table_pages


def find_captions_with_locations(pdf_path):
    captions = []
    potential_blocks = []

    # Step 1: Broadly identify potential caption blocks
    for page_layout in extract_pages(pdf_path, laparams=LAParams()):
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                text = element.get_text()
                # Look for the presence of key phrases, numbering patterns, or table captions including the new generalized pattern
                if (re.search(r'\d+(\.\d+)*\s+Summary of evidence and', text, re.IGNORECASE) or
                        "Summary of evidence and" in text or
                        re.search(r'\d+\.\d+\.\d+(\.\d+)?', text) or
                        re.match(r'Table\s+\d+\.\d+', text) or
                        re.match(r'Table\s+\d+:', text)):
                    potential_blocks.append((text, element.y0, page_layout.pageid))

    # Step 2: Refine and extract captions from potential blocks
    for block, y0, pageid in potential_blocks:
        # Split the block into lines for more granular processing
        lines = block.split('\n')
        for line in lines:
            # Check each line for the target patterns, including the new generalized pattern
            if (re.search(r'\d+(\.\d+)*\s+Summary of evidence and', line,
                          re.IGNORECASE) or "Summary of evidence and guidelines" in line or re.match(
                    r'Table\s+\d+\.\d+', line) or re.match(r'Table\s+\d+:', line)):
                caption = line.strip()  # Clean up the line to serve as the caption
                captions.append((caption, y0, pageid))
                break  # Assuming one primary caption per block; adjust if needed

    return captions


def associate_captions_with_tables(captions, tables):
    caption_table_pairs = []

    for table in tables:
        page_number = table.page
        table_top = table._bbox[3]  # Top coordinate of the table
        page_captions = {}

        i = 0
        for cap in captions:
            if cap[2] == page_number:
                page_captions.update({i: {"cap": cap[0], "dist": cap[1]}})
                i += 1
            elif cap[2] == page_number - 1:
                page_captions.update({i: {"cap": cap[0], "dist": cap[1] + 420}})
                i += 1

        closest_caption = None
        min_distance = float('inf')

        for caption in page_captions:
            distance = abs(page_captions[caption]["dist"] - table_top)
            if distance < min_distance:
                closest_caption = page_captions[caption]
                min_distance = distance

        if closest_caption:
            caption_table_pairs.append((closest_caption["cap"], table.df))

    return caption_table_pairs


def merge_dataframes_with_same_caption(list_of_tuples):
    merged_dict = {}
    # Iterate through the list of tuples
    for caption, df in list_of_tuples:
        # If the caption is already in the dictionary, concatenate the current dataframe with the existing one
        if caption in merged_dict:
            merged_dict[caption] = pd.concat([merged_dict[caption], df], ignore_index=True)
        else:
            merged_dict[caption] = df

    # Convert the dictionary back to a list of tuples
    merged_list_of_tuples = [(caption, df) for caption, df in merged_dict.items()]

    return merged_list_of_tuples


def add_captions_as_rows(list_of_tuples):
    result_list = []

    for caption, df in list_of_tuples:
        # Create a new dataframe with the caption row
        if caption is not None:
            caption_df = pd.DataFrame([caption], columns=[df.columns[0]])
            # Fill remaining columns with empty strings
            for col in df.columns[1:]:
                caption_df[col] = ""

            # Concatenate the caption dataframe with the original dataframe
            # Reset index to avoid index duplication
            new_df = pd.concat([caption_df, df], ignore_index=True)
            result_list.append(new_df)
        else:
            result_list.append(df)

    return result_list


def find_nearest_caption(page, table_top, last_caption):
    pattern = r'\d+(\.\d+)+\s+[A-Za-z]+.*'
    min_distance = float('inf')
    nearest_caption = ""

    for block in page.get_text("blocks"):
        block_text = block[4].strip()
        if re.match(pattern, block_text):
            block_bottom = block[3]
            distance = table_top - block_bottom
            if 0 < distance < min_distance:
                min_distance = distance
                nearest_caption = block_text

    return nearest_caption if nearest_caption else last_caption


def extract_and_filter_tables_with_captions(pdf_path, tables, headings=["Summary of evidence", "Recommendations"]):
    doc = fitz.open(pdf_path)
    filtered_tables_with_captions = []
    last_caption = None  # Initialize last_caption as None

    for table in tables:
        if any(heading in table.df.iloc[0, 0] for heading in headings):
            page_num = table.page - 1
            page = doc.load_page(page_num)
            table_top_edge = table._bbox[1]

            caption = find_nearest_caption(page, table_top_edge, last_caption)
            last_caption = caption  # Update last_caption with the current caption for future use

            filtered_tables_with_captions.append((caption, table.df))

    return filtered_tables_with_captions


def dataframe_to_markdown(df):
    # Check if the first row can be used as the header or if it's a description
    if df.shape[1] > 1:
        if (pd.isna(df.iloc[0, 1]) or df.iloc[0, 1] == '') and df.iloc[0, 0]:
            description = df.iloc[0, 0].strip()  # Store the description, removing any leading/trailing whitespace
            df = df.drop(0).reset_index(drop=True)  # Remove the description row
        else:
            description = None
    else:
        description = df.iloc[0]  # Store the description, removing any leading/trailing whitespace
        df = df.drop(0).reset_index(drop=True)  # Remove the description row

    # Set the first row with entries as headers
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)

    # Convert DataFrame to Markdown
    markdown_table = df.to_markdown(index=False)

    # Prepend description if it exists
    if description is not None:
        markdown_table = f"{description}\n\n{markdown_table}"

    return markdown_table


def get_relevant_pages(pdf_path):
    # Refined regex patterns for specific patterns
    intro_pattern = re.compile(r"\sINTRODUCTION\b")
    refs_pattern = re.compile(r"\sREFERENCES\b")

    intro_count = 0
    refs_count = 0
    start_page = None
    end_page = None

    # Initialize variables to store text of each page
    text_pages = extract_text(pdf_path).split("\f")

    # Iterate through each page's text
    for i, page_text in enumerate(text_pages):
        # Check for the second occurrence of the Introduction pattern
        if intro_pattern.search(page_text):
            intro_count += 1
            if intro_count == 2:
                start_page = i + 1  # We use i+1 since indices are zero-based

        # Check for the second occurrence of the References pattern
        if refs_pattern.search(page_text):
            refs_count += 1
            if refs_count == 2:
                end_page = i
                break  # No need to continue if we found both

    # Return the range of pages between Introduction and References
    if start_page is not None and end_page is not None:
        return start_page, end_page + 1
    else:
        raise ValueError("Couldn't find the specified sections twice in the document.")


def extract_tables_and_captions_with_pdfminer(pdf_path, config):
    captions = find_captions_with_locations(pdf_path)
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

    caption_table_pairs = associate_captions_with_tables(captions, tables)
    merged = merge_dataframes_with_same_caption(caption_table_pairs)
    captioned_tables = add_captions_as_rows(merged)

    evidence_recommendations_tables = extract_and_filter_tables_with_captions(pdf_path, tables)
    evidence_recommendations_tables = add_captions_as_rows(evidence_recommendations_tables)

    cleaned_tables = []
    table_text = []
    dfs = []
    if config["markdown_tables"]:
        evidence_recommendations_tables = [(dataframe_to_markdown(df), df) for df in evidence_recommendations_tables]
        captioned_tables_md = [(dataframe_to_markdown(df), df) for df in captioned_tables]

        table_text += [t[0] for t in evidence_recommendations_tables]
        dfs += [t[1] for t in evidence_recommendations_tables]

        table_text += [t[0] for t in captioned_tables_md]
        dfs += [t[1] for t in captioned_tables_md]

    else:
        for df in evidence_recommendations_tables:
            cleaned_tables.append(prepare_all_text_data(df, separator=config["separator"], clean=False))

        for df in captioned_tables:
            cleaned_tables.append(prepare_all_text_data(df, separator=config["separator"], clean=False))
        for table in cleaned_tables:
            table_text.append(aggregate_table_to_text(table))

    return table_text, dfs
