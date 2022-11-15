import streamlit as st
from annotated_text import annotation
import streamlit.components.v1 as components
import time
import pandas as pd
import requests
import base64
import webbrowser
import ast
import difflib as dif
import pickle
from colorama import Fore, Back, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from ansi2html import Ansi2HTMLConverter
conv = Ansi2HTMLConverter()
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import os
from datetime import datetime
import pythainlp
from pythainlp import sent_tokenize, word_tokenize
import numpy as np
import math
import re
import itertools
from pythainlp.corpus import thai_stopwords
import statistics
from statistics import mode, StatisticsError
from ria import ria
from ast import literal_eval

# TFIDF Ranking
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize
from string import punctuation
import numpy as np

def reset(df):
    cols = df.columns
    return df.reset_index()[cols]
def Sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2
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

def query(query_text, vectorizer, document_matrix, df):
    
    # Transform our string using the vocabulary
    transformed = vectorizer.transform([query_text])
    query = transformed[0:1]

    np_result = np.array(find_similar(query, document_matrix, len(df)))

    mask = np_result[:, 1] > 0 

    np_result = np_result[mask, :]
        
    index_list = []
    score_list = []
    df_dict = {'index': index_list, 'score': score_list}

    df_dict['index'] =  np_result[:,0]
    df_dict['score'] =  np_result[:,1]

    df = pd.DataFrame(df_dict)
    df['index'] = df['index'].astype(int)
    df = df.set_index('index')
    
    return reset(pd.merge(df, data, left_index=True, right_index=True, how ='left'))


# A function that given an input query item returns the top-k most similar items 
# by their cosine similarity.
def find_similar(query_vector, td_matrix, top_k = 5):
    cosine_similarities = cosine_similarity(query_vector, td_matrix).flatten()
    related_doc_indices = cosine_similarities.argsort()[::-1]
    return [(index, cosine_similarities[index]) for index in related_doc_indices][0:top_k]

st.set_page_config(layout="wide", page_title = 'RIA', page_icon = 'fav.png')
st.markdown(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

def link_card(id_val, source, context, pdf_html, doc_meta, doc_meta_2,filter_meta,filter_meta_2,filter_meta_3):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title"><a href="http://pc140032646.bot.or.th/th_ria?code_id={source.split(' ')[0]}" class="card-link">{source}</a></h5>
            <h6>{doc_meta}</h6>
            <h6>{doc_meta_2}</h6>
            <h6>{filter_meta}</h6>
            <h6>{filter_meta_2}</h6>
            <h6>{filter_meta_3}</h6>
            <p class="card-text">{context}</p>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_2(source, source_2, context, pdf_html, doc_no):
# def card_2(source, source_2, context, pdf_html,filter_meta,filter_meta_2,filter_meta_3):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h3 class="card-title">{source}</h3>
            <h3 class="card-title">{doc_no}</h3>
            <h3 class="card-title">{source_2}</h3>
            <h3 class="card-title">เนื้อหา</h3>
            <p class="card-text">{context}</p>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_4(source, context, pdf_html, doc_meta):
# def card_4(source, context, pdf_html, doc_meta,filter_meta,filter_meta_2,filter_meta_3):
    #<div class="card text-white bg-dark mb-3" style="margin:1rem;">
    st.markdown(f"""
        <div class="card" style="margin:1rem;">
            <div class="card-body h-100">
                <h5 class="card-title">{source}</h5>
                <h6>{doc_meta}</h6>
                <p class="card-text">{context}</p>
                {pdf_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

def filter_col(row,row_names,selected_filter_lists):
        result_each_col = []
        y = []
        count_filter = 0
        #กรณีไม่มี Filter ก็แสดงหมด
        if len(selected_filter_lists[0]) + len(selected_filter_lists[1]) + len(selected_filter_lists[2]) == 0:
            return 1
        else:
            for row_number in range(len(row_names)):
                row_name = row_names[row_number]
                selected_filter_list = selected_filter_lists[row_number]
                if len(selected_filter_list) == 0:
                        continue
                for i in selected_filter_list:
                    try:
                        data_in_row = ast.literal_eval(row[row_name])
                    except:
                        data_in_row = row[row_name]
                        pass
                    if i.replace(' ','') in [j.replace(' ','') for j in data_in_row]:
                        result_each_col.append(True)
                    else:
                        result_each_col.append(False)
                    count_filter += 1
            if sum(result_each_col) == count_filter:
                return 1
            else:
                return 0
def filter_result_search(filter1_selected, filter2_selected, filter3_selected, Result_search):
    Result_search["Check"] = Result_search.apply(filter_col,
                                                 args=(['สถาบันผู้เกี่ยวข้อง','ประเภทเอกสาร','กฎหมาย'],
                                                       [filter1_selected,filter2_selected,filter3_selected]),
                                                 axis=1)
    Result_search = Result_search[Result_search['Check'] == 1].copy()
    Result_search = Result_search.drop(columns=['Check'])
    Result_search = Result_search.sort_values(by=['Score'], ascending=False).reset_index(drop=True)
    return Result_search

# @st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def draw_network(HtmlFile):
    components.html(HtmlFile.read(), height = 500)
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    # return df.to_csv().encode('utf-8')
    return df.to_csv(index = False).encode('utf-8-sig')

@st.cache(allow_output_mutation=True)
def get_answer(sentence_query, context_list):
    json_params = {}
    json_params['question'] = sentence_query
    json_params['context'] = context_list
    res = requests.post('http://localhost:6101/qa_pipeline',
                        json = json_params
                    )
    return res.json()

app = ria()
data = app.df_meta_question_answer.copy()
with open('09_Output_Streamlib/tfidf_vectorizer.pickle', 'rb') as handle:
    tfidf_vectorizer = pickle.load(handle)
with open('09_Output_Streamlib/tfidf_term_document_matrix.pickle', 'rb') as handle:
    tfidf_term_document_matrix = pickle.load(handle)


get_params = st.experimental_get_query_params()
# st.markdown(get_params)
if get_params == {}:
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
    st.write("""# RIA Live Demo""")
    c11, c12, c13, c14 = st.columns((6, 2, 4, 4))
    with c11:
        sentence_query = st.text_input('ใส่ข้อความเพื่อค้นหา', key = "sentence_query", placeholder = "การจัดชั้นและการกันเงินสำรอง")
        st.markdown("""หมายเหตุ: สามารถค้นหาเอกสารที่มีหลาย Keyword ที่สำคัญได้ผ่านการใช้ "(keyword1 หรือ keyword2)" เช่น (ความเสี่ยงด้านเครดิต หรือ ความเสี่ยงด้านปฏิบัติการ)""")
        query_params = st.experimental_get_query_params()
        try:
            # http://localhost:8501/?doc_meta=0002|0030|0028
            query_option = query_params['doc_meta'][0]
            st.markdown(query_option)
        except:
            pass
    with c12:
        show_result_type = st.radio(
            "Show Result:",
            ('Distinct Documents', 'All'), key = "show_result_type")

    if 'filter_1' not in st.session_state:
        st.session_state['filter_1'] = []
    if 'filter_2' not in st.session_state:
        st.session_state['filter_2'] = []

    if sentence_query: # or query != '' :
        # Save logs
        current_time = str(datetime.now())[:19]
        if os.path.exists(os.getcwd() + '/log.csv') == False:
            pd.DataFrame([current_time,sentence_query]).T.to_csv('log.csv', index = False)
        else:
            pd.DataFrame([current_time,sentence_query]).T.to_csv('log.csv', mode='a', index=False, header=False)

        # Init State Sessioin
        if 'page' not in st.session_state:
            st.session_state['page'] = 1

        # app.query = sentence_query

        # try:
        # ori_res_df = app.step1_user_search()
        url_query = 'http://127.0.0.1:6102/ria_query'
        post_query_para = {'user_query':sentence_query,'filter1_selected':[],'filter2_selected':[],'filter3_selected':[]}
        post_query = requests.post(url_query,json= post_query_para)
        ori_res_df = pd.DataFrame(post_query.json()['Result_search']) #dataframe
        html_graph = post_query.json()['Network_Graph']
        option_filter = post_query.json()['option_filter']
        filter1_from_result = option_filter['filter1_from_result']
        filter2_from_result = option_filter['filter2_from_result']
        filter3_from_result = option_filter['filter3_from_result']
        # st.dataframe(ori_res_df)

        ori_res_df['Number_result'] = ori_res_df['Number_result'].fillna(-1)
        
        if show_result_type == 'Distinct Documents':
            res_df_01 = ori_res_df.copy()
            res_df_01 = res_df_01.groupby('Doc_ID').first().reset_index()
            res_df_01 = reset(res_df_01.sort_values(by = 'Score', ascending = False))
        else:
            res_df_01 = ori_res_df.copy()

        with c13:
            # filter1_from_result, filter2_from_result, filter3_from_result = app.option_filter(res_df_01)
            filter_1 = st.multiselect(
                'สถาบันการเงินผู้เกี่ยวข้อง:',
                options = filter1_from_result,
                default = [],
                key = 'filter_1',
            )
        
        with c14:
            filter_2 = st.multiselect(
                'ประเภทเอกสาร:',
                options = filter2_from_result,
                default = [],
                key = 'filter_2',
            )
            
        # app.filter1_selected, app.filter2_selected, app.filter3_selected = st.session_state['filter_1'], st.session_state['filter_2'], []
        res_df_02 = filter_result_search(st.session_state['filter_1'], st.session_state['filter_2'], [], res_df_01)

        try:
            with st.spinner("Loading..."):
                # G = app.create_network(res_df_02)
                # G.show('nx.html')
                # HtmlFile = open('nx.html','r',encoding='utf-8')
                # draw_network(HtmlFile)
                with open('nx.html','w') as f:
                    f.write(html_graph)
                HtmlFile = open('nx.html','r',encoding='utf-8')
                draw_network(HtmlFile)
                st.markdown("""<div align="center"><h3>ความเชื่อมโยงประกาศ</h3></div>""", unsafe_allow_html=True)
        except StatisticsError:
            pass

        c21, c22 = st.columns((14, 6))
        with c21:
            res_df = res_df_02.copy()
            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1

            doc_df = res_df.copy()
            doc_df = doc_df[['Doc_ID', 'เรื่อง','File_Code']].drop_duplicates()
            doc_df['sort_id'] = doc_df['Doc_ID'].astype(int)
            doc_df = reset(doc_df.sort_values(by = 'sort_id'))
            
            if len(res_df) > 0:
                st.session_state['max_page'] = res_df['page'].max()
                filter_res_df = reset(res_df[res_df['page'] == st.session_state['page']])
                filter_res_df['สถาบันผู้เกี่ยวข้อง'] = filter_res_df['สถาบันผู้เกี่ยวข้อง'].apply(literal_eval)
                filter_res_df['ประเภทเอกสาร'] = filter_res_df['ประเภทเอกสาร'].apply(literal_eval)
                filter_res_df['กฎหมาย'] = filter_res_df['กฎหมาย'].apply(literal_eval)

                # st.dataframe(filter_res_df)

                for i in range(len(filter_res_df)):
                    content = filter_res_df['Original_text'].values[i]
                    doc_name = filter_res_df['เรื่อง'].values[i]
                    doc_meta = filter_res_df['Doc_Page_ID'].values[i]
                    content = app.highlight_text(sentence_query, content)
                    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a> <a href='#linkto_top' class="card-link">Link to top</a> <a href='#linkto_bottom' class="card-link">Link to bottom</a>""".format(filter_res_df['File_Code'].values[i])
                    
                    context = '...{}...'.format(content)
                    doc_meta = 'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name
                    doc_meta_2 = 'Page ID: {}'.format(filter_res_df['Doc_Page_ID'].values[i].split('|')[1])
                    filter_meta = 'สถาบันผู้เกี่ยวข้อง: ' + ' | '.join(filter_res_df['สถาบันผู้เกี่ยวข้อง'].values[i])
                    filter_meta_2 = 'ประเภทเอกสาร: ' + ' | '.join(filter_res_df['ประเภทเอกสาร'].values[i])
                    filter_meta_3 = 'กฎหมายที่เกี่ยวข้อง: ' + ' | '.join(filter_res_df['กฎหมาย'].values[i])
                    doc_no = 'ประกาศเลขที่: {}'.format(filter_res_df['เลขที่ (Thai)'].values[i])
                    
                    if filter_res_df['Number_result'].values[i] > 0:
                        source = doc_meta.replace('|','|Page') + ' (Click to See This Page)'
                        st.markdown(f"""
                        <div class="card" style="margin:1rem;">
                            <div class="card-body">
                                <h5 class="card-title"><a href="http://pc140032646.bot.or.th/th_ria?code_id={filter_res_df['Doc_Page_ID'].values[i]}" class="card-link">{source}</a></h5>
                                <h6>{doc_no}</h6>
                                <h6>{doc_meta_2}</h6>
                                <h6>{filter_meta}</h6>
                                <h6>{filter_meta_2}</h6>
                                <h6>{filter_meta_3}</h6>
                                <p class="card-text">{context}</p>
                                {pdf_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        source = doc_meta.replace('|','|Page')
                        st.markdown(f"""
                        <div class="card" style="margin:1rem;">
                            <div class="card-body">
                                <h5 class="card-title">{source}</h5>
                                <h6>{doc_no}</h6>
                                <h6>{doc_meta_2}</h6>
                                <h6>{filter_meta}</h6>
                                <h6>{filter_meta_2}</h6>
                                <h6>{filter_meta_3}</h6>
                                <p class="card-text">{context}</p>
                                {pdf_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                cols = ['Doc_Page_ID','เรื่อง','Original_text']
                csv = convert_df(res_df[cols])

        with c22:
            st.markdown("""Remark:\n- Scroll เพื่อซุมเข้าออก\n- Click เพื่อเลื่อน""")
            markdown_text = "### List of Documents"
            for index, row in doc_df.iterrows():
                markdown_text = markdown_text + """\n[{}: {}](http://pc140032646.bot.or.th/th_pdf/{})\n""".format(row['Doc_ID'], row['เรื่อง'], row['File_Code'])
            st.markdown(markdown_text)

        if 'max_page' not in st.session_state:
            st.session_state['max_page'] = 10
        c41, c42 = st.columns((14, 6))
        with c41:
            st.markdown("<div id='linkto_bottom'></div>", unsafe_allow_html=True)
            if int(st.session_state['max_page']) > 1:
                page = st.slider('Page No:', 1, int(st.session_state['max_page']), key = 'page')
            st.download_button(
                label="Download search results as CSV",
                data=csv,
                file_name=f"{sentence_query}_results.csv",
                mime='text/csv',
            )
            st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
        # except:
        #     st.markdown("## ไม่พบข้อความที่ค้นหา")
        #     pass

elif 'code_id' in get_params:
    code_id = get_params['code_id'][0]
    doc_meta = code_id.replace('Doc','').replace('Page','')
    # part_one_df = app.part_one_show_original_text(doc_meta)
    # st.dataframe(part_one_df)

    url_compare = 'http://127.0.0.1:6102/ria_compare'
    post_compare = requests.post(url_compare,json= {'Doc_Page_ID':doc_meta})
    part_one_df = pd.DataFrame(post_compare.json()['Result_original'])

    doc_name = part_one_df['เรื่อง'].values[0]
    content = part_one_df['Original_text'].values[0]
    file_name = part_one_df['File_Code'].values[0]

    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(file_name)
    card_2(
        'Doc{} '.format(doc_meta.split('|')[0]) + doc_name, 
        'Page{}'.format(doc_meta.split('|')[1]),
        '...{}...'.format(content),
        pdf_html,
        'ประกาศเลขที่: {}'.format(part_one_df['เลขที่ (Thai)'].values[0])
        # 'ทด',
        # 'ทด',
        # 'ทด',
    )

    # part_two_df = app.part_two_show_compare(doc_meta)
    # st.dataframe(part_two_df)
    part_two_df = pd.DataFrame(post_compare.json()['Result_compare'])

    st.markdown("<br>", unsafe_allow_html=True)
    for index in range(len(part_two_df)):
        doc_id = 'Doc' + part_two_df['Q_Doc_ID'].values[index]
        page_id = "Page" + part_two_df['Q_Page_ID'].values[index] + ' Sentence'  + part_two_df['Q_Sen_ID'].values[index]
        result_sentence = part_two_df['query_Sentence_show'].values[index]
        pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(part_two_df['Q_File_Code'].values[index])
        left_title = doc_id + ' ' + part_two_df['Q_เรื่อง'].values[index]
        left_sen = conv.convert(result_sentence)
        left_page_id = page_id
        left_pdf = pdf_html
        left_doc_no = "ประกาศเลขที่: {}".format(part_two_df['Q_เลขที่ (Thai)'].values[index])

        doc_id = 'Doc' + part_two_df['R_Doc_ID'].values[index]
        page_id = "Page" + part_two_df['R_Page_ID'].values[index] + ' Sentence'  + part_two_df['R_Sen_ID'].values[index]
        result_sentence = part_two_df['result_Sentence_show'].values[index]
        pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(part_two_df['R_File_Code'].values[index])
        right_title = doc_id + ' ' + part_two_df['R_เรื่อง'].values[index]
        right_sen = conv.convert(result_sentence)
        right_page_id = page_id
        right_pdf = pdf_html
        right_doc_no = "ประกาศเลขที่: {}".format(part_two_df['R_เลขที่ (Thai)'].values[index])

        st.markdown(f""" 
            <div class="card-deck">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">{left_title}</h5>
                        <h6>{left_doc_no}</h6>
                        <h6>{left_page_id}</h6>
                        <p class="card-text">{left_sen}</p>
                        {left_pdf}
                    </div>
                </div>
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">{right_title}</h5>
                        <h6>{right_doc_no}</h6>
                        <h6>{right_page_id}</h6>
                        <p class="card-text">{right_sen}</p>
                        {right_pdf}
                    </div>
                </div>
            </div>
            <br>
        """, unsafe_allow_html=True)

elif 'qa' in get_params:
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
    st.write("""# QA RIA Live Demo""")
    c11, c12, c13, c14 = st.columns((6, 2, 4, 4))
    with c11:
        sentence_query = st.text_input('ใส่ข้อความเพื่อค้นหา', key = "sentence_query", placeholder = "การจัดชั้นและการกันเงินสำรอง")
        st.markdown("""หมายเหตุ: สามารถค้นหาเอกสารที่มีหลาย Keyword ที่สำคัญได้ผ่านการใช้ "(keyword1 หรือ keyword2)" เช่น (ความเสี่ยงด้านเครดิต หรือ ความเสี่ยงด้านปฏิบัติการ)""")
        query_params = st.experimental_get_query_params()
        try:
            # http://localhost:8501/?doc_meta=0002|0030|0028
            query_option = query_params['doc_meta'][0]
            # st.markdown(query_option)
        except:
            pass
    with c12:
        show_result_type = st.radio(
            "Show Result:",
            ('All', 'Distinct Documents'), key = "show_result_type")

    # if 'filter_1' not in st.session_state:
    #     st.session_state['filter_1'] = []
    # if 'filter_2' not in st.session_state:
    #     st.session_state['filter_2'] = []

    if sentence_query: # or query != '' :
        # Save logs
        current_time = str(datetime.now())[:19]
        if os.path.exists(os.getcwd() + '/log.csv') == False:
            pd.DataFrame([current_time,sentence_query]).T.to_csv('log.csv', index = False)
        else:
            pd.DataFrame([current_time,sentence_query]).T.to_csv('log.csv', mode='a', index=False, header=False)

        # Init State Sessioin
        if 'page' not in st.session_state:
            st.session_state['page'] = 1

        app.query = sentence_query

        query_text = sentence_query
        ori_res_df = query(query_text, tfidf_vectorizer, tfidf_term_document_matrix, data)
        ori_res_df = ori_res_df.head(100)
        # st.dataframe(ori_res_df)

        with st.spinner("QA Processing..."):
            res_list = get_answer(sentence_query, list(ori_res_df['Original_text'].values))

        score_list = [each['score'] for each in res_list]
        ans_list = [each['answer'] for each in res_list]
        start_list = [each['start'] for each in res_list]
        end_list = [each['end'] for each in res_list]
        ori_res_df['answer'] = ans_list
        ori_res_df['answer_score'] = score_list
        ori_res_df['start'] = start_list
        ori_res_df['end'] = end_list

        # ori_res_df['Number_result'] = ori_res_df['Number_result'].fillna(-1)

        if show_result_type == 'Distinct Documents':
            res_df_01 = ori_res_df.copy()
            res_df_01 = res_df_01.groupby('Doc_ID').first().reset_index()
            res_df_01 = reset(res_df_01.sort_values(by = 'answer_score', ascending = False))
        else:
            res_df_01 = ori_res_df.copy()

        with c13:
            filter1_from_result, filter2_from_result, filter3_from_result = app.option_filter(res_df_01)
            filter_1 = st.multiselect(
                'สถาบันการเงินผู้เกี่ยวข้อง:',
                options = filter1_from_result,
                default = [],
                key = 'filter_1',
            )
        
        with c14:
            filter_2 = st.multiselect(
                'ประเภทเอกสาร:',
                options = filter2_from_result,
                default = [],
                key = 'filter_2',
            )
        # st.markdown(st.session_state['show_result_type'])
        # st.markdown(st.session_state['filter_1'])
        # st.markdown(st.session_state['filter_2'])
        app.filter1_selected, app.filter2_selected, app.filter3_selected = st.session_state['filter_1'], st.session_state['filter_2'], []
        # res_df_02 = reset(app.filter_result_search(res_df_01))

        res_df_02 = res_df_01.copy()
        res_df_02 = reset(res_df_02.sort_values(by=['answer_score'], ascending = False))

        c21, c22 = st.columns((14, 6))
        with c21:
            res_df = res_df_02.copy()
            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1

            doc_df = res_df.copy()
            doc_df = doc_df[['Doc_ID', 'เรื่อง','File_Code']].drop_duplicates()
            doc_df['sort_id'] = doc_df['Doc_ID'].astype(int)
            doc_df = reset(doc_df.sort_values(by = 'sort_id'))
            
            if len(res_df) > 0:
                st.session_state['max_page'] = res_df['page'].max()
                filter_res_df = reset(res_df[res_df['page'] == st.session_state['page']])
                filter_res_df['สถาบันผู้เกี่ยวข้อง'] = filter_res_df['สถาบันผู้เกี่ยวข้อง'].apply(literal_eval)
                filter_res_df['ประเภทเอกสาร'] = filter_res_df['ประเภทเอกสาร'].apply(literal_eval)
                filter_res_df['กฎหมาย'] = filter_res_df['กฎหมาย'].apply(literal_eval)

                for i in range(len(filter_res_df)):
                    content = filter_res_df['Original_text'].values[i]
                    doc_name = filter_res_df['เรื่อง'].values[i]
                    doc_meta = filter_res_df['Doc_Page_ID'].values[i]

                    answer = filter_res_df['answer'].values[i]
                    answer_score = filter_res_df['answer_score'].values[i]
                    highlight_text = filter_res_df['Original_text'].values[i][filter_res_df['start'].values[i]:filter_res_df['end'].values[i]]
                    content = content.replace(highlight_text, f"""<mark style="background-color:yellow;">{highlight_text}</mark>""")

                    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a> <a href='#linkto_top' class="card-link">Link to top</a> <a href='#linkto_bottom' class="card-link">Link to bottom</a>""".format(filter_res_df['File_Code'].values[i])
                    # if filter_res_df['Number_result'].values[i] > 0:
                    row_1 = 'Doc' + doc_meta.replace('|','|Page') + ' (Click to See This Page)'
                    row_2 = 'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name
                    row_3 = 'Page ID: {}'.format(doc_meta.split('|')[1])
                    row_4 = 'สถาบันผู้เกี่ยวข้อง: ' + ' | '.join(filter_res_df['สถาบันผู้เกี่ยวข้อง'].values[i])
                    row_5 = 'ประเภทเอกสาร: ' + ' | '.join(filter_res_df['ประเภทเอกสาร'].values[i])
                    row_6 = 'กฎหมายที่เกี่ยวข้อง: ' + ' | '.join(filter_res_df['กฎหมาย'].values[i])
                    row_7 = 'Score: {}'.format(filter_res_df['answer_score'].values[i])
                    content = '...{}...'.format(content)
                    answer = 'คำตอบ: {}'.format(answer)

                    st.markdown(f"""
                    <div class="card" style="margin:1rem;">
                        <div class="card-body">
                            <h6>{answer}</h6>
                            <h5 class="card-title"><a href="http://pc140032646.bot.or.th/th_ria?code_id={row_1.split(' ')[0]}" class="card-link">{row_1}</a></h5>
                            <h6>{row_2}</h6>
                            <h6>{row_3}</h6>
                            <h6>{row_4}</h6>
                            <h6>{row_5}</h6>
                            <h6>{row_6}</h6>
                            <h6>{row_7}</h6>
                            <p class="card-text">{content}</p>
                            {pdf_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # else:
                    #     row_1 = 'Doc' + doc_meta.replace('|','|Page')
                    #     row_2 = 'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name
                    #     row_3 = 'Page ID: {}'.format(doc_meta.split('|')[1])
                    #     row_4 = 'สถาบันผู้เกี่ยวข้อง: ' + ' | '.join(filter_res_df['สถาบันผู้เกี่ยวข้อง'].values[i])
                    #     row_5 = 'ประเภทเอกสาร: ' + ' | '.join(filter_res_df['ประเภทเอกสาร'].values[i])
                    #     row_6 = 'กฎหมายที่เกี่ยวข้อง: ' + ' | '.join(filter_res_df['กฎหมาย'].values[i])
                    #     row_7 = 'Score: {}'.format(filter_res_df['answer_score'].values[i])
                    #     content = '...{}...'.format(content)
                    #     answer = 'คำตอบ: {}'.format(answer)

                    #     st.markdown(f"""
                    #     <div class="card" style="margin:1rem;">
                    #         <div class="card-body">
                    #             <h6>{answer}</h6>
                    #             <h5 class="card-title">{row_1}</h5>
                    #             <h6>{row_2}</h6>
                    #             <h6>{row_3}</h6>
                    #             <h6>{row_4}</h6>
                    #             <h6>{row_5}</h6>
                    #             <h6>{row_6}</h6>
                    #             <h6>{row_7}</h6>
                    #             <p class="card-text">{content}</p>
                    #             {pdf_html}
                    #         </div>
                    #     </div>
                    #     """, unsafe_allow_html=True)
                cols = ['Doc_Page_ID','เรื่อง','Original_text']
                csv = convert_df(res_df[cols])

        if 'max_page' not in st.session_state:
            st.session_state['max_page'] = 10
        c41, c42 = st.columns((14, 6))
        with c41:
            st.markdown("<div id='linkto_bottom'></div>", unsafe_allow_html=True)
            if int(st.session_state['max_page']) > 1:
                page = st.slider('Page No:', 1, int(st.session_state['max_page']), key = 'page')
            st.download_button(
                label="Download search results as CSV",
                data=csv,
                file_name=f"{sentence_query}_results.csv",
                mime='text/csv',
            )
            st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
        # except:
        #     st.markdown("## ไม่พบข้อความที่ค้นหา")
        #     pass