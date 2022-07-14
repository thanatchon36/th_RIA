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

path = 'source'
with open(f'{path}/dict_doc_page_search.pickle', 'rb') as file:
    dict_doc_page_search = pickle.load(file)
with open(f'{path}/tfidf_vectorizer.sav', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open(f'{path}/tfidf_term_document_matrix.sav', 'rb') as file:
    tfidf_term_document_matrix = pickle.load(file)
code_doc_mapping = pd.read_csv(f'{path}/code_doc_mapping.csv',dtype=str)
code_mapping = code_doc_mapping.copy()
data_max_pair = pd.read_csv(f'{path}/data_max_pair4.csv')
code_doc_mapping_dict = dict(zip(code_doc_mapping.doc_id, code_doc_mapping.name))
with open(f'{path}/dict_query.pickle', 'rb') as file:
    dict_query = pickle.load(file)
with open(f'{path}/dict_pair.pickle', 'rb') as file:
    dict_pair = pickle.load(file)
with open(f'{path}/dict_query_list.pickle', 'rb') as file:
    dict_query_list = pickle.load(file)

#ใช้แค่ TF เพียงอย่างเดียว
def search(word_query,search_type,tfidf_term_document_matrix):
    def exact_word_search(word_query):
        doc_id_list = []
        page_id_list = []
        doc_page_id_list = []
        count_word = []
        for key, value in dict_doc_page_search.items():
            doc_page_id_list.append(key)
            doc_id_list.append(key.split('|')[0])
            page_id_list.append(key.split('|')[1])
            count_word.append(value.count(word_query))
        doc_list = [dict_doc_page_search[x] for x in list(dict_doc_page_search.keys()) if x in doc_page_id_list]
        df_search = pd.DataFrame(data={'score':count_word,'doc_id':doc_page_id_list,'doc_list':doc_list})
        return_df = df_search[df_search['score']!=0].sort_values(by=['score','doc_id'],ascending=False)
        return return_df

    def any_words_search(word_query,tfidf_term_document_matrix):
        index_list = []
        doc_id_list = []
        doc_list = []
        for index,doc in enumerate(list(dict_doc_page_search.values())):
            doc_id_list.append(list(dict_doc_page_search.keys())[index])
            index_list.append(index)
            doc_list.append(doc)
        dict_query_df = pd.DataFrame(data={'index':index_list,'doc_id':doc_id_list,'doc_list':doc_list})
        transformed = tfidf_vectorizer.transform([word_query])
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
        df = df.drop(columns='index')
        df = df[df['score'] >0]
        return df

    if search_type == 1:
        df_return = exact_word_search(word_query)
    elif search_type == 2:
            df_return = any_words_search(word_query,tfidf_term_document_matrix)
    df_query = df_return.copy()
    df_query['show_doc_id'] = df_query['doc_id'].apply(lambda x: x.split('|')[0])
    df_query['doc_name'] = df_query['show_doc_id'].apply(lambda x: code_doc_mapping_dict[x])
    df_query['page_id'] = df_query['doc_id'].apply(lambda x: x.split('|')[1])
    df_query['doc_detail'] = df_query['doc_list']
    return reset(df_query).head(100)

def create_network(df_query,data_max_pair):
    if df_query.shape[0] != 0:
        doc_id_query_network = []
        for code_id in df_query['doc_id']:
            doc_id_query_network.append(code_id.split("|")[0])
        doc_id_query_network = list(set(doc_id_query_network))
    else:
        doc_id_query_network = None
        
    doc_id_A_list = []
    doc_id_B_list = []
    doc_id_list = []
    set_of_pair_list = []
    
    code_id_pairs = list(data_max_pair['max_pair_list'])
    for code_id_pair in code_id_pairs:
        code_id_split = code_id_pair.split(' ~ ')
        doc_id_A = code_id_split[0].split('|')[0]
        doc_id_B = code_id_split[1].split('|')[0]
        set_of_pair_list.append({doc_id_A,doc_id_B})
    
    if doc_id_query_network != None:
        doc_id_list.extend(doc_id_query_network)
    else:
        doc_id_list = []
    
    G = Network()
    for doc_A in doc_id_list:
        for doc_B in doc_id_list:
            G.add_node(doc_A)
            G.add_node(doc_B)
            if {doc_A,doc_B} in set_of_pair_list:
                weight = set_of_pair_list.count({doc_A,doc_B})
                G.add_edge(doc_A, doc_B, value=weight)
    return G

st.set_page_config(layout="wide", page_title = 'RIA', page_icon = 'fav.png')
st.markdown(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

def card(id_val, source, context, pdf_html, doc_meta, doc_meta_2):
    #<div class="card text-white bg-dark mb-3" style="margin:1rem;">
    #<h5 class="card-title"><a href="http://localhost:8602/th_ria_explorer/?doc_meta={source}" class="card-link">{source}</a></h5>
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title"><a href="http://pc140032646.bot.or.th/th_ria?code_id={source.split(' ')[0]}" class="card-link">{source}</a></h5>
            <h6>{doc_meta}</h6>
            <h6>{doc_meta_2}</h6>
            <p class="card-text">{context}</p>
            <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_2(source, source_2, context, pdf_html):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h3 class="card-title">{source}</h3>
            <h3 class="card-title">{source_2}</h3>
            <h3 class="card-title">เนื้อหา</h3>
            <p class="card-text">{context}</p>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_3(source, context, pdf_html, doc_meta, doc_meta_2, doc_meta_3):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title"><a href="http://pc140032646.bot.or.th/th_ria/?doc_meta={source.split(' ')[0]}" class="card-link">{source}</a></h5>
            <h6>{doc_meta}</h6>
            <h6>{doc_meta_2}</h6>
            <h6>{doc_meta_3}</h6>
            <p class="card-text">{context}</p>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_4(id_val, source, context, pdf_html, doc_meta):
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

#2_app
def step2_user_click_doc(doc_id_query):
    key_dict_query = []
    value_dict_query = []
    for doc_id in list(dict_query.keys()):
        if doc_id_query in doc_id:
            key_dict_query.append(doc_id)
            value_dict_query.append(dict_query[doc_id])
    df_pair_result = pd.DataFrame(data={'key_dict_query':key_dict_query,'value_dict_query':value_dict_query})
    df_pair_result['code_id'] = df_pair_result['key_dict_query']
    df_pair_result['doc_id'] = df_pair_result['key_dict_query'].apply(lambda x: x.split('|')[0])
    df_pair_result['page_id'] = df_pair_result['key_dict_query'].apply(lambda x: x.split('|')[1])
    df_pair_result['sentence_id'] = df_pair_result['key_dict_query'].apply(lambda x: x.split('|')[2])
    df_pair_result['sentence_detail'] = df_pair_result['value_dict_query']
    df_pair_result['doc_name'] = df_pair_result['doc_id'].apply(lambda x: code_doc_mapping_dict[x])
    return df_pair_result

#3_app
def click_query(query_code_id):
    index_query = dict_pair['query'].index(query_code_id)
    index_match_query = dict_pair['result'][index_query]
    index_match_score = dict_pair['Score'][index_query]
    result_sentence_list = []
    for result_from_query in index_match_query:
        query_sentence, result_sentence = show_result(query_code_id,result_from_query)
        result_sentence_list.append(result_sentence)
    return query_sentence, result_sentence_list, index_match_query, index_match_score
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
def get_document_info(query_code_id):
    query_code_id_split = query_code_id.split("|")
    doc_id = query_code_id_split[0]
    page_id = query_code_id_split[1]
    sentence_id = query_code_id_split[2]
    doc_name = code_mapping[code_mapping['doc_id'] == doc_id].iloc[0]['name']
    return doc_id, page_id, sentence_id, doc_name

# @st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def draw_network(HtmlFile):
    components.html(HtmlFile.read(), height = 540)

get_params = st.experimental_get_query_params()
# st.markdown(get_params)
if get_params == {}:
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
    st.write("""# RIA Live Demo""")
    c11, c12, c13 = st.columns((14, 3, 3))
    with c11:
        sentence_query = st.text_input('ค้นหาภาษาไทย', key = "sentence_query", placeholder = "พ.ร.บ. ธุรกิจสถาบันการเงิน")
        query_params = st.experimental_get_query_params()
        try:
            # http://localhost:8501/?doc_meta=0002|0030|0028
            query_option = query_params['doc_meta'][0]
            st.markdown(query_option)
        except:
            pass
    with c12:
        search_type = st.radio(
            "Find pages with:",
            ('This exact word or phrase', 'Any of these words'), key = "search_type")
    with c13:
        show_result_type = st.radio(
            "Show Result:",
            ('All', 'Distinct'), key = "show_result_type")

    if sentence_query: # or query != '' :
        # Init State Sessioin
        if 'page' not in st.session_state:
            st.session_state['page'] = 1
        c21, c22 = st.columns((14, 6))

        with c21:
            if search_type == 'This exact word or phrase':
                search_type_val = 1
            else:
                search_type_val = 2
            res_df = search(sentence_query,search_type_val,tfidf_term_document_matrix)

            # st.dataframe(res_df)
            if show_result_type == 'Distinct':
                res_df = res_df.groupby('show_doc_id').first().reset_index()
                res_df = reset(res_df.sort_values(by = 'score', ascending = False))

            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1

            doc_df = res_df.copy()
            doc_df = doc_df[['show_doc_id', 'doc_name']].drop_duplicates()
            doc_df['sort_id'] = doc_df['show_doc_id'].astype(int)
            doc_df = reset(doc_df.sort_values(by = 'sort_id'))
            
            # st.dataframe(res_df)
            # st.dataframe(doc_df)

            if len(res_df) > 0:
                st.session_state['max_page'] = res_df['page'].max()
                filter_res_df = res_df[res_df['page'] == st.session_state['page']]
                for i in range(len(filter_res_df)):
                    score = round(filter_res_df['score'].values[i], 2)
                    content = filter_res_df['doc_detail'].values[i]
                    doc_name = filter_res_df['doc_name'].values[i]
                    doc_meta = filter_res_df['doc_id'].values[i]
                    for each_j in get_found_token(st.session_state['sentence_query'], content):
                        content = content.replace(each_j, f"<mark>{each_j}</mark>")
                    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a> <a href='#linkto_top' class="card-link">Link to top</a> <a href='#linkto_bottom' class="card-link">Link to bottom</a>""".format(doc_meta.split('|')[0] + '.pdf')
                    card('Relevance: {}'.format(score), 
                        doc_meta + ' (Click to See This Page)',
                        '...{}...'.format(content),
                        pdf_html,
                        'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name,
                        'Page ID: {}'.format(doc_meta.split('|')[1]),
                    )
        with c22:
            st.markdown("""<div align="center"><h3>ความเชื่อมโยงประกาศ</h3></div>""", unsafe_allow_html=True)
            with st.spinner("Loading..."):
                G = create_network(res_df, data_max_pair)
                G.show('nx.html')
                HtmlFile = open('nx.html','r',encoding='utf-8')
                draw_network(HtmlFile)
            st.markdown("""Remark:\n- Scroll เพื่อซุมเข้าออก\n- Click เพื่อเลื่อน""")
            markdown_text = "### List of Documents"
            for index, row in doc_df.iterrows():
                markdown_text = markdown_text + """\n[{}: {}](http://pc140032646.bot.or.th/th_pdf/{}.pdf)\n""".format(row['show_doc_id'], row['doc_name'], row['show_doc_id'])
            st.markdown(markdown_text)

        if 'max_page' not in st.session_state:
            st.session_state['max_page'] = 10
        c41, c42 = st.columns((14, 6))
        with c41:
            st.markdown("<div id='linkto_bottom'></div>", unsafe_allow_html=True)
            if int(st.session_state['max_page']) > 1:
                page = st.slider('Page No:', 1, int(st.session_state['max_page']), key = 'page')
                st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
elif 'code_id' in get_params:
    code_id = get_params['code_id'][0]
    doc_meta = code_id
    doc_name = code_doc_mapping_dict[doc_meta.split('|')[0]]
    content = dict_doc_page_search[code_id]
    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(doc_meta.split('|')[0] + '.pdf')
    card_2(
        'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name, 
        'Page ID: {}'.format(doc_meta.split('|')[1]),
        '...{}...'.format(content),
        pdf_html,
    )

    df_pair_result = step2_user_click_doc(code_id)
    filter_res_df = df_pair_result.copy()
    if len(filter_res_df) > 0:
        st.markdown("### ประโยคที่มีความเชื่อมโยงกับประกาศอื่นๆ", unsafe_allow_html=True)
        
        # st.dataframe(df_pair_result)
        for i in range(len(filter_res_df)):
            score = ""
            content = filter_res_df['sentence_detail'].values[i]
            doc_name = filter_res_df['doc_name'].values[i]
            doc_meta = filter_res_df['code_id'].values[i]
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(doc_meta.split('|')[0] + '.pdf')
            card_3( 
                doc_meta + ' (Click to See This Page)',
                '...{}...'.format(content),
                pdf_html,
                'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name,
                'Page ID: {}'.format(doc_meta.split('|')[1]),
                'Sentence ID: {}'.format(doc_meta.split('|')[2]),
            )

elif 'doc_meta' in get_params:
    c01, c02 = st.columns((8, 2))
    with c01:
        st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
        st.write("""# RIA Explorer""")

    c11, c12 = st.columns((12, 2))
    with c11:
        query_params = get_params
        try:
            # http://localhost:8501/?doc_meta=0002|0030|0028
            query_id = query_params['doc_meta'][0]
        except:
            pass
        query_sentence, result_sentence_list, index_match_query, index_match_score = click_query(query_id)
        result_sentence_list = result_sentence_list[:10]
        doc_id, page_id, sentence_id, doc_name = get_document_info(query_id)
        pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(doc_id + '.pdf')
        card_4("", 
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
                card_4("", 
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
                card_4("", 
                    doc_id,
                    '{}'.format(conv.convert(result_sentence)),
                    pdf_html,
                    doc_name,
                )