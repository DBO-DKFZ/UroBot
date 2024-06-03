import textwrap
import markdown

import pandas as pd

from openai import OpenAI
from chromadb import PersistentClient
from embedding import SentenceTransformerEmbeddingFunction
from flask import Flask, request, render_template_string, render_template, jsonify

app = Flask(__name__)


# Mock-up of initializing your database client and other components as necessary
def initialize_components():
    global openai_client, embedding_func, db_client, collection
    openai_client = OpenAI()

    embedding_func = SentenceTransformerEmbeddingFunction()
    embedding_func.initialize_model()

    db_client = PersistentClient(path="UroBot_database")
    collection = db_client.get_collection(name="UroBot_v1.0", embedding_function=embedding_func)


initialize_components()


def convert_markdown_to_html_or_text(input_text):
    lines = input_text.strip().split('\n')
    output = ""
    inside_table = False
    table_started = False
    alignments = []
    html_table = ''

    for i, line in enumerate(lines):
        # Detect the start of a table by looking for a header and a separator
        if (not table_started and '|' in line and i + 1 < len(lines) and
                '|' in lines[i + 1] and all(c in '|:- ' for c in lines[i + 1].strip())):
            if not inside_table:
                # There might be text before the table starts
                if output.strip():
                    output += "<p>" + output.strip() + "</p>\n"
                output += '<table>\n'
                inside_table = True
                table_started = True
                html_table = '  <tr>\n'
            continue
        elif table_started and line.strip() == "":
            # End of table detected
            output += html_table + '</table>\n'
            inside_table = False
            table_started = False
            alignments = []
            continue

        if inside_table:
            if table_started and all(c in '|:- ' for c in line.strip()):
                # This is a header separator line, set alignments
                alignments = [
                    'center' if cell.strip().startswith(':') and cell.strip().endswith(':') else
                    'right' if cell.strip().endswith(':') else
                    'left' for cell in line.strip('|').split('|')
                ]
                table_started = False  # Stop header processing
                continue
            # Process normal row
            cells = line.strip('|').split('|')
            cell_tag = 'th' if table_started else 'td'
            for idx, cell in enumerate(cells):
                align_style = f' style="text-align: {alignments[idx]};"' if alignments else ''
                html_table += f'    <{cell_tag}{align_style}>{cell.strip()}</{cell_tag}>\n'
            html_table += '  </tr>\n'

        else:
            if output.strip():
                output += line + "\n"
            else:
                output = line + "\n"

    # Final check to close any open table
    if inside_table:
        output += html_table + '</table>\n'

    return output.strip()


def process_query(query):
    query_results = collection.query(query_texts=[query], n_results=9)
    context = ""

    documents = []
    for i, item in enumerate(query_results["documents"][0]):
        id = query_results["ids"][0][i]
        context += f"\nDocument ID {id[2:]}:\n{item}\n"
        if query_results["metadatas"][0][i]["paragraph_type"] == "table":
            df = pd.read_csv(query_results["metadatas"][0][i]["dataframe"]).to_html(index=False)
            documents.append(f"Document ID {id[2:]}:\n \n{df} \n")
        else:
            documents.append(f"Document ID {id[2:]}:\n \n{convert_markdown_to_html_or_text(item)} \n")

    updated_query = "You are a helpful and understanding urologist answering questions to the patient." \
                    f" Use full sentences and answer human-like and aks if you can answer more questions after" \
                    f" giving an answer based on the following context: \n" \
                    f"---" \
                    f"{context}" \
                    f"--- \n" \
                    f"If the context does not provide information on the question respond with 'Sorry my knowledge base does not include information on that topic'" \
                    f"Ensure your answer is annotated with the Document IDs of the context that were used to answer the question. " \
                    f"Make sure you use the following format for the annotations: (Document ID 'number_given_in_context')." \
                    f" You must use the words Document ID for each annotation."

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": updated_query},
            {"role": "user", "content": query}
        ],
        temperature=0.2,
        max_tokens=2000
    )

    return completion.choices[0].message.content, documents


@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    query = None
    documents = None
    if request.method == 'POST':
        query = request.form['query']
        answer, documents = process_query(query)

    return render_template('index.html', answer=answer, query=query, documents=documents)


if __name__ == '__main__':
    app.run(debug=True)
