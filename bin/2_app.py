import streamlit as st
from annotated_text import annotation
import streamlit.components.v1 as components
import time
import pandas as pd
import numpy as np
import requests
import base64
import webbrowser
import ast 
import difflib as dif
import pickle
from colorama import Fore, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize
from string import punctuation
from ansi2html import Ansi2HTMLConverter
conv = Ansi2HTMLConverter()
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def reset(df):
    cols = df.columns
    return df.reset_index()[cols]
def norm_token(token):
    token = token.lower()
    return token
def get_found_token(query, text):
    token_text_list = word_tokenize(text)
    token_query_list = word_tokenize(query)
    norm_token_text_list = [norm_token(each) for each in token_text_list]
    norm_query_text_list = [norm_token(each) for each in token_query_list]
    text_dict = {}
    for i, each in enumerate(token_text_list):
        text_dict[i] = each
    found_token_list = []
    for each in norm_query_text_list:
        if each in norm_token_text_list:
            found_token_list.append(each)
    stopword_list = [" "]
    found_token_list = [each for each in found_token_list if each not in stopword_list]
    found_token_list = list(set(found_token_list))
    found_token_index_list = [i for i, each in enumerate(norm_token_text_list) if each in found_token_list]
    found_token_index_res_list = [text_dict[each] for each in found_token_index_list]
    found_token_index_res_list = list(set(found_token_index_res_list))
    return found_token_index_res_list

def canonicalize(string):
    normalized_tokens = list()
    a = word_tokenize(string, engine = 'newmm')
    for j in a:
        token = j.strip()
        #Add clean statement here 
        if len(token) > 1 and token not in set(punctuation) and token not in ['..','...','ๆๆ']:
            try:
                normalized_tokens.append(token.lower())
            except:
                normalized_tokens.append(token)
                pass
    return normalized_tokens
# A function that given an input query item returns the top-k most similar items 
# by their cosine similarity.
def find_similar(query_vector, td_matrix, top_k = 5):
    cosine_similarities = cosine_similarity(query_vector, td_matrix).flatten()
    related_doc_indices = cosine_similarities.argsort()[::-1]
    return [(index, cosine_similarities[index]) for index in related_doc_indices][0:top_k]
code_mapping = pd.read_csv('code_doc_mapping.csv',dtype='str')
with open('dict_query.pickle', 'rb') as file:
    dict_query = pickle.load(file)
with open('dict_query_list.pickle', 'rb') as file:
    dict_query_list = pickle.load(file)
with open('dict_pair.pickle', 'rb') as file:
    dict_pair = pickle.load(file)
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
tfidf_term_document_matrix = pickle.load(open('tfidf_term_document_matrix.sav', 'rb'))
def change_dict_query_to_df(dict_query):
    index_list = []
    code_id_list = []
    sentence_list = []
    for index,doc in enumerate(list(dict_query.values())):
        code_id_list.append(list(dict_query.keys())[index])
        index_list.append(index)
        sentence_list.append(doc)
    dict_query_df = pd.DataFrame(data={'index':index_list,'code_id':code_id_list,'sentence_list':sentence_list})
    return dict_query_df
dict_query_df = change_dict_query_to_df(dict_query)
def query(query_text,tfidf_term_document_matrix,dict_query):
    transformed = tfidf_vectorizer.transform([query_text])
    query = transformed[0:1]
    np_result = np.array(find_similar(query, tfidf_term_document_matrix, len(list(dict_query.values()))))
    index_list = []
    score_list = []
    df_dict = {'index': index_list, 'score': score_list}

    df_dict['index'] =  np_result[:,0]
    df_dict['score'] =  np_result[:,1]

    df = pd.DataFrame(df_dict)
    df['index'] = df['index'].astype(int)
    df = df.merge(dict_query_df,on='index',how='left')
    df = df[df['score'] >0]
    return df
def get_document_info(query_code_id):
    query_code_id_split = query_code_id.split("|")
    doc_id = query_code_id_split[0]
    page_id = query_code_id_split[1]
    sentence_id = query_code_id_split[2]
    doc_name = code_mapping[code_mapping['doc_id'] == doc_id].iloc[0]['name']
    return doc_id, page_id, sentence_id, doc_name
def show_result(query_code_id,mapping_result_id):
    query_sentence = dict_query_list[query_code_id]
    compare_sentence = dict_query_list[mapping_result_id]
    compare_sentence_result_list = list(dif.Differ().compare(query_sentence,compare_sentence))
    new_str1 = ''
    new_str2 = ''
    len_first = 0 #เช็กว่าเป็นคำแรกของประโยคไหม ถ้าเป็นก็จะตัดออก เพื่อปรับให้ประโยคตรงกัน
    for symbol_and_word in compare_sentence_result_list:
        symbol = symbol_and_word[:2]
        word = symbol_and_word[2:]
        if symbol == '+ ' and len_first!= 0:
            new_str1 = new_str1 + f"{Fore.RED}{word}{Fore.BLACK}"
        elif symbol == '- ' and len_first!= 0:
            new_str2 = new_str2 + f"{Fore.BLACK}{word}"
        elif symbol == '? ':
            print('None')
        elif symbol != '+ ' and symbol != '- ' and symbol != '? ':
            len_first += 1
            new_str1 = new_str1 + f"{Fore.BLACK}{word}"
            new_str2 = new_str2 + f"{Fore.BLACK}{word}"
    return new_str2.replace('BLANK',' '),new_str1.replace('BLANK',' ')
def click_query(query_code_id):
    index_query = dict_pair['query'].index(query_code_id)
    index_match_query = dict_pair['result'][index_query]
    index_match_score = dict_pair['Score'][index_query]
    result_sentence_list = []
    for result_from_query in index_match_query:
        query_sentence, result_sentence = show_result(query_code_id,result_from_query)
        result_sentence_list.append(result_sentence)
    return query_sentence, result_sentence_list, index_match_query, index_match_score
def create_network(df_query):
    if df_query.shape[0] != 0:
        doc_id_query = []
        for code_id in df_query['code_id']:
            doc_id_query.append(code_id.split("|")[0])
            doc_id_query = list(set(doc_id_query))
    else:
        doc_id_query = None
    data_max_pair = pd.read_csv('data_max_pair4.csv')
    doc_id_A_list = []
    doc_id_B_list = []
    doc_id_list = []
    set_of_pair_list = []
    code_id_pairs = list(data_max_pair['max_pair_list'])
    for code_id_pair in code_id_pairs:
        code_id_split = code_id_pair.split(' ~ ')
        doc_id_A = code_id_split[0].split('|')[0]
        doc_id_B = code_id_split[1].split('|')[0]
        if doc_id_query != None and (doc_id_A in doc_id_query or doc_id_B in doc_id_query):
            doc_id_A_list.append(doc_id_A)
            doc_id_B_list.append(doc_id_B)
            set_of_pair_list.append({doc_id_A,doc_id_B})
        elif doc_id_query == None:
            doc_id_A_list.append(doc_id_A)
            doc_id_B_list.append(doc_id_B)
            set_of_pair_list.append({doc_id_A,doc_id_B})  
    if len(doc_id_A_list) == 0:
        return None
    doc_id_list.extend(doc_id_A_list)
    doc_id_list.extend(doc_id_B_list)
    doc_id_list = list(set(doc_id_list))
    G = Network()
    for doc_A in doc_id_list:
        for doc_B in doc_id_list:
            if doc_A != doc_B and {doc_A,doc_B} in set_of_pair_list:
                weight = set_of_pair_list.count({doc_A,doc_B})
                G.add_node(doc_A)
                G.add_node(doc_B)
                # G.add_edge(doc_A, doc_B, weight=weight )
                G.add_edge(doc_A, doc_B, value=weight)
    return G

st.set_page_config(layout="wide", page_title = 'RIA Explorer', page_icon = 'fav.png')
st.markdown(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

def card(id_val, source, context, pdf_html, doc_meta):
    #<div class="card text-white bg-dark mb-3" style="margin:1rem;">
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6 class="card-subtitle mb-2 text-muted">{doc_meta}</h6>
            <p class="card-text">{context}</p>
            <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

c01, c02 = st.columns((8, 2))
with c01:
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
    st.write("""# RIA Explorer""")

c11, c12 = st.columns((12, 2))
with c11:
    query_params = st.experimental_get_query_params()
    try:
        # http://localhost:8501/?doc_meta=0002|0030|0028
        query_id = query_params['doc_meta'][0]
    except:
        pass
    query_sentence, result_sentence_list, index_match_query, index_match_score = click_query(query_id)
    result_sentence_list = result_sentence_list[:10]
    doc_id, page_id, sentence_id, doc_name = get_document_info(query_id)
    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(doc_id + '.pdf')
    card("", 
        doc_id,
        '{}'.format(conv.convert(query_sentence)),
        pdf_html,
        doc_name,
    )
    res_df = {
        'doc_id': [],
        'page_id': [],
        'sentence_id': [],
        'doc_name': [],
        'result_sentence': [],
    }
    for index , result_sentence in enumerate(result_sentence_list):
        doc_id, page_id, sentence_id, doc_name = get_document_info(index_match_query[index])
        res_df['doc_id'].append(doc_id)
        res_df['page_id'].append(page_id)
        res_df['sentence_id'].append(sentence_id)
        res_df['doc_name'].append(doc_name)
        res_df['result_sentence'].append(result_sentence)
    res_df = pd.DataFrame(res_df)

    doc_id_list = list(pd.unique(res_df['doc_id'].values))
    st.markdown("#### Result No: {}".format(len(result_sentence_list)))

with c12:
    filter_result = st.multiselect(
     'Filter:',
     doc_id_list,
     doc_id_list,
     key = 'filter_result',
    )

filter_res_df = reset(res_df[res_df['doc_id'].isin(st.session_state['filter_result'])])
c21, c22 = st.columns((4, 4))
with c21:
    for index in range(len(filter_res_df)):
        if index in [0, 2, 4 , 6, 8, 10]:
            doc_id = filter_res_df['doc_id'].values[index]
            sentence_id = filter_res_df['sentence_id'].values[index]
            doc_name = filter_res_df['doc_name'].values[index]
            result_sentence = filter_res_df['result_sentence'].values[index]
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(doc_id + '.pdf')
            card("", 
                doc_id,
                '{}'.format(conv.convert(result_sentence)),
                pdf_html,
                doc_name,
            )
with c22:
    for index in range(len(filter_res_df)):
        if index in [1, 3, 5, 7, 9, 11]:
            doc_id = filter_res_df['doc_id'].values[index]
            sentence_id = filter_res_df['sentence_id'].values[index]
            doc_name = filter_res_df['doc_name'].values[index]
            result_sentence = filter_res_df['result_sentence'].values[index]
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(doc_id + '.pdf')
            card("", 
                doc_id,
                '{}'.format(conv.convert(result_sentence)),
                pdf_html,
                doc_name,
            )