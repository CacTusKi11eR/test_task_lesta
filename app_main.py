import math
import re
import os
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for
)

app = Flask(__name__)

########### realize in config ###############
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_PATH'] = UPLOAD_FOLDER
app.secret_key = '123'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#############################################

ITEMS_PER_PAGE = 50

def calculate_tf(text):
    '''
    Calculates term frequency for input text.

    parameter: text - input text
    returns: term frequency
    '''
    words = re.findall(r'\b\w+\b', text.lower())
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    return tf

def calculate_idf(word_counts_list, total_documents):
    '''
    Calculates inverse document frequency for input word.
    parameter: word_counts - dictionary with word counts
    parameter: total_documents - total number of documents
    returns: inverse document frequency
    '''
    idf = {}
    all_words = set()
    for word_counts in word_counts_list:
        all_words.update(word_counts.keys())

    for word in all_words:
        document_count = sum(1 for word_counts in word_counts_list if word in word_counts)
        idf[word] = math.log(total_documents / (document_count))
    
    return idf

def process_files(filepaths):
    '''
    Processes a single file, calculates term frequency, inverse document frequency, and returns a DataFrame.

    parameter: filepath - file path of the text file
    returns: DataFrame with term frequency and inverse document frequency
    '''
    word_counts_list = []
    all_texts = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()
        word_counts_list.append(calculate_tf(text))
        all_texts.append(text)
    
    idf_scores = calculate_idf(word_counts_list, len(filepaths))

    aggregated_word_counts = {}
    for word_counts in word_counts_list:
        for word, count in word_counts.items():
            aggregated_word_counts[word] = aggregated_word_counts.get(word, 0) + count

    data = []
    for word, tf in aggregated_word_counts.items():
        data.append({
            'word':word,
            'tf':tf,
            'idf':idf_scores.get(word, 0)
        })
    df = pd.DataFrame(data)
    df = df.sort_values(by='idf', ascending=False)

    return df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    '''
    Uploads a file and processes its contents.

    returns: rendered HTML page with table or error message
    '''
    if request.method == 'POST':
        files = request.files.getlist('files')
        filepaths = []

        if not files or all(file.filename == '' for file in files):
            return render_template('index.html', error='No files selected')

        for file in files:
            if file and file.filename != '':
                filepath = os.path.join(app.config['UPLOAD_PATH'], file.filename)
                file.save(filepath)
                filepaths.append(filepath)

        if filepaths:
            session['filepaths'] = filepaths
            return redirect(url_for('show_results', page=1))
        else:
            return render_template('index.html', error='Error processing files')

    return render_template('index.html')
    
@app.route('/results/page/<int:page>')
def show_results(page):
    """
    Displays the TF-IDF table
    """
    filepaths = session.get('filepaths')
    if not filepaths:
        return redirect(url_for('upload_file'))

    df = process_files(filepaths)

    start_index = (page - 1) * ITEMS_PER_PAGE
    end_index = start_index + ITEMS_PER_PAGE
    df_page = df.iloc[start_index:end_index]

    total_pages = math.ceil(len(df) / ITEMS_PER_PAGE)

    table_html = df_page.to_html(index=False)

    return render_template('index.html', table=table_html, page=page, total_pages=total_pages)


if __name__ == '__main__':
    app.run(debug=True)
