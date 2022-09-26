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

self = ria()

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

st.set_page_config(layout="wide", page_title = 'RIA', page_icon = 'fav.png')
st.markdown(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """, unsafe_allow_html=True)

def card(id_val, source, context, pdf_html, doc_meta, doc_meta_2):
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6>{doc_meta}</h6>
            <h6>{doc_meta_2}</h6>
            <p class="card-text">{context}</p>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def link_card(id_val, source, context, pdf_html, doc_meta, doc_meta_2):
    #<div class="card text-white bg-dark mb-3" style="margin:1rem;">
    #<h5 class="card-title"><a href="http://localhost:8602/th_ria_explorer/?doc_meta={source}" class="card-link">{source}</a></h5>
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title"><a href="http://pc140032646.bot.or.th/th_ria?code_id={source.split(' ')[0]}" class="card-link">{source}</a></h5>
            <h6>{doc_meta}</h6>
            <h6>{doc_meta_2}</h6>
            <p class="card-text">{context}</p>
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

def card_4(source, context, pdf_html, doc_meta):
    #<div class="card text-white bg-dark mb-3" style="margin:1rem;">
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6>{doc_meta}</h6>
            <p class="card-text">{context}</p>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

# @st.cache(suppress_st_warning=True)
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def draw_network(HtmlFile):
    components.html(HtmlFile.read(), height = 500)
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    # return df.to_csv().encode('utf-8')
    return df.to_csv(index = False).encode('utf-8-sig')

get_params = st.experimental_get_query_params()
# st.markdown(get_params)
if get_params == {}:
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
    st.write("""# RIA Live Demo""")
    c11, c12, c13 = st.columns((14, 3, 3))
    with c11:
        sentence_query = st.text_input('ใส่ข้อความเพื่อค้นหา', key = "sentence_query", placeholder = "การจัดชั้นและการกันเงินสำรอง")
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

        self.query = sentence_query

        try:
            res_df = self.step1_user_search()
            Result_search = res_df.copy()
            res_df['Number_result'] = res_df['Number_result'].fillna(-1)

            try:
                with st.spinner("Loading..."):
                    G = self.create_network(Result_search)
                    G.show('nx.html')
                    HtmlFile = open('nx.html','r',encoding='utf-8')
                    draw_network(HtmlFile)
                    st.markdown("""<div align="center"><h3>ความเชื่อมโยงประกาศ</h3></div>""", unsafe_allow_html=True)
            except StatisticsError:
                pass

            c21, c22 = st.columns((14, 6))
            with c21:    
                if show_result_type == 'Distinct Documents':
                    res_df = res_df.groupby('Doc_ID').first().reset_index()
                    res_df = reset(res_df.sort_values(by = 'Score', ascending = False))

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
                    for i in range(len(filter_res_df)):
                        content = filter_res_df['Original_text'].values[i]
                        doc_name = filter_res_df['เรื่อง'].values[i]
                        doc_meta = filter_res_df['Doc_Page_ID'].values[i]

                        # for each_j in get_found_token(st.session_state['sentence_query'], content):
                        #     content = content.replace(each_j, f"<mark>{each_j}</mark>")
                        # content = content.replace(sentence_query, f"""<mark style="background-color:yellow;">{sentence_query}</mark>""")
                        content = self.highlight_text(sentence_query, content)

                        pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a> <a href='#linkto_top' class="card-link">Link to top</a> <a href='#linkto_bottom' class="card-link">Link to bottom</a>""".format(filter_res_df['File_Code'].values[i])
                        if filter_res_df['Number_result'].values[i] > 0:
                            link_card("", 
                                'Doc' + doc_meta.replace('|','|Page') + ' (Click to See This Page)',
                                '...{}...'.format(content),
                                pdf_html,
                                'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name,
                                'Page ID: {}'.format(doc_meta.split('|')[1]),
                            )
                        else:
                            card("", 
                                'Doc' + doc_meta.replace('|','|Page'),
                                '...{}...'.format(content),
                                pdf_html,
                                'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name,
                                'Page ID: {}'.format(doc_meta.split('|')[1]),
                            )
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
        except:
            st.markdown("## ไม่พบข้อความที่ค้นหา")
            pass

elif 'code_id' in get_params:
    code_id = get_params['code_id'][0]
    doc_meta = code_id.replace('Doc','').replace('Page','')
    part_one_df = self.part_one_show_original_text(doc_meta)

    doc_name = part_one_df['เรื่อง'].values[0]
    content = part_one_df['Original_text'].values[0]
    file_name = self.Data_Dictionary_streamlib_0[self.Data_Dictionary_streamlib_0['Doc_ID'] == doc_meta.split('|')[0].replace('Doc','')]['File_Code'].values[0]

    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(file_name)
    card_2(
        'Doc{} '.format(doc_meta.split('|')[0]) + doc_name, 
        'Page{}'.format(doc_meta.split('|')[1]),
        '...{}...'.format(content),
        pdf_html,
    )

    part_two_df = self.part_two_show_compare(doc_meta)
    # st.dataframe(part_two_df)

    c21, c22 = st.columns((4, 4))
    with c21:
        for index in range(len(part_two_df)):
            doc_id = 'Doc' + part_two_df['Q_Doc_ID'].values[index]
            page_id = "Page" + part_two_df['Q_Page_ID'].values[index] + ' Sentence'  + part_two_df['Q_Sen_ID'].values[index]
            result_sentence = part_two_df['query_Sentence_show'].values[index]
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(self.Data_Dictionary_streamlib_0[self.Data_Dictionary_streamlib_0['Doc_ID'] == part_two_df['Q_Doc_ID'].values[index]]['File_Code'].values[0])
            card_4( 
                doc_id + ' ' + part_two_df['Q_เรื่อง'].values[index],
                '{}'.format(conv.convert(result_sentence)),
                pdf_html,
                page_id,
            )

    with c22:
        for index in range(len(part_two_df)):
            doc_id = 'Doc' + part_two_df['R_Doc_ID'].values[index]
            page_id = "Page" + part_two_df['R_Page_ID'].values[index] + ' Sentence'  + part_two_df['R_Sen_ID'].values[index]
            result_sentence = part_two_df['result_Sentence_show'].values[index]
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(self.Data_Dictionary_streamlib_0[self.Data_Dictionary_streamlib_0['Doc_ID'] == part_two_df['R_Doc_ID'].values[index]]['File_Code'].values[0])
            card_4( 
                doc_id + ' ' + part_two_df['R_เรื่อง'].values[index],
                '{}'.format(conv.convert(result_sentence)),
                pdf_html,
                page_id,
            )