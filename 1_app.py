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

# File for Network
df_dict_pair_0 = pd.read_csv('09_Output_Streamlib/df_dict_pair.csv')
Data_Dictionary_streamlib_0 = pd.read_csv('09_Output_Streamlib/Data_Dictionary_streamlib.csv',dtype=str)

# File for Search
Doc_Page_Text_1 = pd.read_csv('09_Output_Streamlib/P_One_Doc_Page_Text.csv')
category_text_1 = pd.read_csv('09_Output_Streamlib/category_text_score.csv')
Data_Dictionary_streamlib_1 = pd.read_csv('09_Output_Streamlib/Data_Dictionary_streamlib.csv',dtype=str)
df_dict_pair_1 = pd.read_csv('09_Output_Streamlib/df_dict_pair.csv')

# File For Compare
df_dict_pair_2 = pd.read_csv('09_Output_Streamlib/df_dict_pair.csv')
Data_Dictionary_streamlib_2 = pd.read_csv('09_Output_Streamlib/Data_Dictionary_streamlib.csv',dtype=str)
Doc_Page_Text_2 = pd.read_csv('09_Output_Streamlib/P_One_Doc_Page_Text.csv')
Doc_Page_Sentence_2 = pd.read_csv('09_Output_Streamlib/Doc_Page_Sentence.csv')

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

def create_query_list(query):
    open_bracket_location = []
    close_bracket_location = []
    for bracket in re.finditer('{', query):
        open_bracket_location.append(bracket.span()[1])
    for bracket in re.finditer('}', query):
        close_bracket_location.append(bracket.span()[0])
    pair_bracket_location = []
    if len(open_bracket_location) == len(close_bracket_location):
        for index in range(len(open_bracket_location)):
            pair_bracket_location.append((open_bracket_location[index],close_bracket_location[index]))
            
    query_split_all = []
    for pair in pair_bracket_location:
        query_splits = query[pair[0]:pair[1]].split('|')
        query_split_all.append(query_splits)
        
    reword_list = []
    for index in range(len(pair_bracket_location)-1, -1, -1):
        pair_bracket = pair_bracket_location[index]
        replace_word = query[pair_bracket[0]-1:pair_bracket[1]+1]
        reword_list.append(f'REWORD_{index}')
        query = query.replace(replace_word,f'REWORD_{index}')
    reword_list = sorted(reword_list)

    query_list = []
    replcae_query = query
    for element in itertools.product(*query_split_all):
        replcae_query = query
        for index_element in range(len(element)):
            replcae_query = replcae_query.replace(reword_list[index_element],element[index_element])
        query_list.append(replcae_query)
    return query_list

def create_query_token(query):
    query_token = word_tokenize(query)
    stopwords = list(thai_stopwords())
    query_token = [i for i in query_token if i not in stopwords]
    return query_token

def find_token_location_in_doc(document,query_token):
    token_location_all = []
    for token in query_token:
        location_token_list = []
        for location_token in re.finditer(token, document):
             location_token_list.append(location_token.span()[1])
        token_location_all.append(location_token_list)
    return token_location_all

#ไม่สามารถค้นหาคำเดียวได้จึงต้องใช้find_candidate_df_for_len_one
def find_candidate_df(token_location_all):
    candidate_element_list = []
    for element in itertools.product(*token_location_all):
        i = 1
        condition_list = []
        diff_location_element = []
        while i < len(element):
            if element[i] > element[i-1]:
                condition_list.append(True)
                diff_location = element[i] - element[i-1]
                diff_location_element.append(diff_location)
            i = i + 1
        if all(condition_list) and len(condition_list) == len(element)-1:
            add_element = (element) + (sum(diff_location_element)/len(diff_location_element),)
            candidate_element_list.append(add_element)
    df = pd.DataFrame(candidate_element_list)
    df = df.rename(columns={df.columns[-1]:'score'})
    #df = df.groupby(list(df.columns)[:-1])[list(df.columns)[-1:][0]].agg('min').reset_index().drop_duplicates(subset=0,keep='first')
    return df

def find_candidate_df_for_len_one(token_location_all):
    df = pd.DataFrame(data={0:token_location_all[0],'score':[1 for i in range(0,len(token_location_all[0]))]})
    return df

def find_min_location_token(document,query_token):
    token_location_all = find_token_location_in_doc(document,query_token)
    if len(query_token) > 1:
        df = find_candidate_df(token_location_all)
        candidate_df = find_candidate_df(token_location_all)
        for column in df.columns[:-1]:
            df_group = df.groupby(by=[column])['score'].agg('min').reset_index()
            df = df.merge(df_group,how='inner', on=[column,'score'])
    else:
        df = find_candidate_df_for_len_one(token_location_all)
        candidate_df = find_candidate_df_for_len_one(token_location_all)
    return df,candidate_df

def retrieval_score(document,query):
    query_list = create_query_list(query)
    df = pd.DataFrame()
    candidate_df = pd.DataFrame()
    for query in query_list:
        query_token = create_query_token(query)
        query_token = list(filter(lambda token: token != ' ', query_token))
        try:
            df_,candidate_df = find_min_location_token(document,query_token)
            df = pd.concat([df,df_])
        except:
            continue
    try:
        word_columns = [col_word for col_word in candidate_df.columns if col_word!= 'score']
        count = 0
        median_score = statistics.median(df['score'])
        for word in word_columns:
            count += len(df[word].unique())
        tf_score = count/len(word_columns)
        retrieval = tf_score+(tf_score/median_score)
        return retrieval
    except:
        return 0

def create_category_score(category_text,query):
    query_list = create_query_list(query)
    all_query_token = list(set([token for token in create_query_token(query) for query in query_list]))
    try:
        all_query_token.remove('|')
        all_query_token.remove('{')
        all_query_token.remove('}')
    except ValueError:
        pass
    filter_col = list(filter(lambda col: col in query_list , category_text.columns[1:]))
    filter_col.append('cat')
    df_score = pd.DataFrame(data = {'Category_Code':category_text['cat'],'Cat_Score':category_text[filter_col].sum(axis = 1)})
    df_score = df_score.sort_values(by='Cat_Score',ascending=False)
    df_score['rank'] = [str(i) for i in range(len(df_score)-1,-1,-1)]
    df_score = df_score.reset_index(drop=True)
    return df_score

def filter_node_for_search(df_dict_pair,Result_search):
    Result_search_unique = Result_search['Doc_Page_ID'].unique()
    df_dict_pair_filter = df_dict_pair[df_dict_pair['Doc_Page_ID'].isin(Result_search_unique)]
    return df_dict_pair_filter

def step1_user_search(query,Doc_Page_Text,category_text,df_dict_pair,Data_Dictionary_streamlib):
    Doc_Page_Text['Score'] = Doc_Page_Text.apply(lambda x: retrieval_score(x['Original_text'], query), axis=1)
    Result_search = Doc_Page_Text.sort_values(by='Score', ascending=False)
    Result_search = Result_search[Result_search['Score'] > 0]
    Result_search['Score'] = Result_search['Score'].round(3)
    category_score = create_category_score(category_text,query)
    category_score = category_score.astype({'Category_Code': 'int'}).astype({'Category_Code': 'str'})
    Result_search[['Doc_ID', 'Page_ID']] = Result_search['Doc_Page_ID'].str.split('|', 1, expand=True)
    Result_search = Result_search.merge(Data_Dictionary_streamlib,on='Doc_ID',how='left')
    Result_search = Result_search.merge(category_score,on='Category_Code',how='left')
    Result_search = Result_search.sort_values(by=['Score','rank'],ascending=False)
    Result_search = Result_search.drop(columns=['File_Name','Cat_Score','rank'])
    df_dict_pair[['Q_Doc_ID','Q_Page_ID','Q_Sen_ID']] = df_dict_pair['query'].str.split('|', expand=True)
    df_dict_pair[['R_Doc_ID','R_Page_ID','R_Sen_ID']] = df_dict_pair['result'].str.split('|', expand=True)
    df_dict_pair['Doc_Page_ID'] = df_dict_pair['Q_Doc_ID'] + '|' + df_dict_pair['Q_Page_ID'] 
    df_dict_pair_filter_node = filter_node_for_search(df_dict_pair,Result_search).groupby('Doc_Page_ID')['result'].agg('count').reset_index().rename(columns={'result':'Number_result'})
    Result_search = Result_search.merge(df_dict_pair_filter_node,on='Doc_Page_ID',how='left')
    return Result_search

def filter_node(df_dict_pair,Result_search):
    Result_search_unique = Result_search['Doc_Page_ID'].unique()
    df_dict_pair_filter = df_dict_pair[df_dict_pair['Doc_Page_ID'].isin(Result_search_unique)]
    return df_dict_pair_filter

#500 * 2800
def create_network(df_dict_pair,Data_Dictionary_streamlib,Result_search):
    df_dict_pair[['Q_Doc_ID','Q_Page_ID','Q_Sen_ID']] = df_dict_pair['query'].str.split('|', expand=True)
    df_dict_pair[['R_Doc_ID','R_Page_ID','R_Sen_ID']] = df_dict_pair['result'].str.split('|', expand=True)
    df_dict_pair['Doc_Page_ID'] = df_dict_pair['Q_Doc_ID'] + '|' + df_dict_pair['Q_Page_ID'] 
    df_dict_pair_filter = filter_node(df_dict_pair,Result_search)
    all_pair_Doc_id = df_dict_pair_filter[['Q_Doc_ID','R_Doc_ID']].copy()
    all_pair_Doc_id['Count'] = 1
    all_pair_Doc_id_group = all_pair_Doc_id.groupby(['Q_Doc_ID','R_Doc_ID'])['Count'].agg('count').reset_index()
    #print(all_pair_Doc_id_group)
    median_score = statistics.median(all_pair_Doc_id_group['Count'])
    all_node = list(all_pair_Doc_id_group['Q_Doc_ID'].unique())
    all_node.extend(all_pair_Doc_id_group['R_Doc_ID'].unique())
    all_node = list(set(all_node))
    G = Network(height='500px', width='100%',bgcolor="#222222",font_color="white")
    for Doc_ID in all_node:
        Doc_Name = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เรื่อง'].iloc[0]
        Doc_ID_Name_len = len(Doc_Name)
        if Doc_ID_Name_len > 100:
            Doc_Name = Doc_Name[:round(Doc_ID_Name_len/2)]+'\n'+Doc_Name[round(Doc_ID_Name_len/2):]
        G.add_node(Doc_ID,title=[Doc_ID+' :\n'+Doc_Name],shape='circle')
    try:
        for Q_Doc_ID in all_pair_Doc_id_group['Q_Doc_ID'].unique():
            Number_connect_nodes = len(all_pair_Doc_id_group[all_pair_Doc_id_group['Q_Doc_ID'] == Q_Doc_ID])
            for Number_connect_node in range(Number_connect_nodes):
                R_Doc_ID = all_pair_Doc_id_group[all_pair_Doc_id_group['Q_Doc_ID'] == Q_Doc_ID]['R_Doc_ID'].iloc[Number_connect_node]
                weight = all_pair_Doc_id_group[all_pair_Doc_id_group['Q_Doc_ID'] == Q_Doc_ID]['Count'].iloc[Number_connect_node]
                if weight > median_score:
                    weight = 8
                elif weight < median_score:
                    weight = 4
                else:
                    weight = 6
                G.add_edge(str(Q_Doc_ID), str(R_Doc_ID), value=str(weight),title='จำนวนคู่ที่เหมือนกัน:'+str(weight))
        neighbor_map = G.get_adj_list()               
        for node in G.nodes:
            node['title'][0] += '\n\n Neighbors:\n'
            #node['size'] = node['size'] = len(neighbor_map[node['id']])*2
            #print(node,node['size'])
            for Doc_ID in neighbor_map[node['id']]:
                Doc_ID_Name = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เรื่อง'].iloc[0]
                Doc_ID_Name_len = len(Doc_ID_Name)
                if Doc_ID_Name_len > 100:
                    node['title'][0] += f' {Doc_ID} :'+Doc_ID_Name[:round(Doc_ID_Name_len/2)]+'\n'+Doc_ID_Name[round(Doc_ID_Name_len/2):]+ '\n'
                else:
                    node['title'][0] += f' {Doc_ID} :'+Doc_ID_Name+ '\n'
    except:
        pass
    G.set_options('''
var options = {
  "nodes": {
    "color": {
      "border": "rgba(34, 42, 89,1)",
      "background": "rgba(11, 81, 89,1)",
      "highlight": {
        "border": "rgba(242, 76, 39,1)",
        "background": "rgba(242, 234, 194,1)"
      },
      "hover": {
        "border": "rgba(242, 234, 194,1)",
        "background": "rgba(242, 76, 39,1)"
      }
    }
  },
  "edges": {
    "color": {
      "hover": "rgba(217, 181, 4,1)",
      "inherit": false
    },
    "font": {
      "align": "middle"
    },
    "hoverWidth": 3.1,
    "smooth": false
  },
  "interaction": {
    "hover": true
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -59050,
      "centralGravity": 1.5,
      "springLength": 45,
      "springConstant": 0.001
    },
    "minVelocity": 0.75
  }
}
''')
    return G

def part_one_show_original_text(Doc_Page_Text,Data_Dictionary_streamlib,Doc_Page_ID):
    Data_Dictionary_streamlib = Data_Dictionary_streamlib[['Doc_ID','เรื่อง']].copy()
    Doc_Page_Text[['Doc_ID','Page_ID']] = Doc_Page_Text['Doc_Page_ID'].str.split('|', expand=True)
    df_part_one = Doc_Page_Text[Doc_Page_Text['Doc_Page_ID'] == Doc_Page_ID].merge(Data_Dictionary_streamlib,on='Doc_ID',how='left')
    return df_part_one

def step3_2_click_show_result(query_sentence,compare_sentence):    
    compare_sentence_result_list = list(dif.Differ().compare(query_sentence,compare_sentence))
    
    new_str1 = ''
    new_str2 = ''
    len_first = 0 #เช็กว่าเป็นคำแรกของประโยคไหม ถ้าเป็นก็จะตัดออก เพื่อปรับให้ประโยคตรงกัน
    
    new_query_sentence = ''
    new_compare_sentence = ''
    for symbol in compare_sentence_result_list:
        if symbol[0] == ' ':
            new_query_sentence += symbol[2:]
            new_compare_sentence += symbol[2:]
        elif symbol[0] == '-':
            new_query_sentence += f"{Fore.BLUE}{symbol[2:]}{Fore.BLACK}" # bleu
        elif symbol[0] == '+':
            new_compare_sentence += f"{Fore.RED}{symbol[2:]}{Fore.BLACK}" # Red
    
    return [new_query_sentence.replace('BLANK',' '),new_compare_sentence.replace('BLANK',' ')]

def part_two_show_compare(df_dict_pair,Doc_Page_Sentence,Data_Dictionary_streamlib,Doc_Page_ID):
    Data_Dictionary_streamlib = Data_Dictionary_streamlib[['Doc_ID','เรื่อง']].copy()
    df_dict_pair[['Q_Doc_ID','Q_Page_ID','Q_Sen_ID']] = df_dict_pair['query'].str.split('|', expand=True)
    df_dict_pair[['R_Doc_ID','R_Page_ID','R_Sen_ID']] = df_dict_pair['result'].str.split('|', expand=True)
    df_dict_pair['Doc_Page_ID'] = df_dict_pair['Q_Doc_ID'] +'|'+df_dict_pair['Q_Page_ID']
    df_dict_pair_filter = df_dict_pair[df_dict_pair['Doc_Page_ID'] == Doc_Page_ID].copy()
    df_dict_pair_filter = df_dict_pair_filter.merge(Doc_Page_Sentence,right_on = 'Doc_Page_Sen_ID',left_on='query',how='left').drop(columns='Doc_Page_Sen_ID').rename(columns={'Sentence':'query_Sentence'})
    df_dict_pair_filter = df_dict_pair_filter.merge(Doc_Page_Sentence,right_on = 'Doc_Page_Sen_ID',left_on='result',how='left').drop(columns='Doc_Page_Sen_ID').rename(columns={'Sentence':'result_Sentence'})
    df_dict_pair_filter['All_Compare'] = df_dict_pair_filter.apply(lambda x: step3_2_click_show_result(x.query_Sentence, x.result_Sentence), axis=1)
    split_df = pd.DataFrame(df_dict_pair_filter['All_Compare'].tolist(), columns=['query_Sentence_show','result_Sentence_show'])
    result_all = pd.concat([df_dict_pair_filter,split_df], axis=1)
    result_all = result_all.drop(columns=['query_Sentence','result_Sentence','All_Compare']).sort_values(by='query')
    result_all = result_all.merge(Data_Dictionary_streamlib,right_on='Doc_ID',left_on='Q_Doc_ID',how='left')
    result_all = result_all.rename(columns={'เรื่อง':'Q_เรื่อง'}).drop(columns=['Doc_ID'])
    result_all = result_all.merge(Data_Dictionary_streamlib,right_on='Doc_ID',left_on='R_Doc_ID',how='left')
    result_all = result_all.rename(columns={'เรื่อง':'R_เรื่อง'}).drop(columns=['Doc_ID'])
    return result_all

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

        res_df = step1_user_search(sentence_query,Doc_Page_Text_1,category_text_1,df_dict_pair_1,Data_Dictionary_streamlib_1)
        Result_search = res_df.copy()

        res_df['Number_result'] = res_df['Number_result'].fillna(-1)
        # st.dataframe(res_df)
        
        # c01, c02, c03 = st.columns((1, 30, 1))
        # with c02:
        try:
            with st.spinner("Loading..."):
                G = create_network(df_dict_pair_0,Data_Dictionary_streamlib_0,Result_search)
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
                    content = content.replace(sentence_query, f"""<mark style="background-color:yellow;">{sentence_query}</mark>""")
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
                            'Doc' + doc_meta.replace('|','|Page') + ' (Click to See This Page)',
                            '...{}...'.format(content),
                            pdf_html,
                            'Document ID: {} '.format(doc_meta.split('|')[0]) + doc_name,
                            'Page ID: {}'.format(doc_meta.split('|')[1]),
                        )
                # st.dataframe(res_df)
                # st.dataframe(doc_df)

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

elif 'code_id' in get_params:
    code_id = get_params['code_id'][0]
    doc_meta = code_id.replace('Doc','').replace('Page','')
    part_one_df = part_one_show_original_text(Doc_Page_Text_2,Data_Dictionary_streamlib_2,doc_meta)

    doc_name = part_one_df['เรื่อง'].values[0]
    content = part_one_df['Original_text'].values[0]
    file_name = Data_Dictionary_streamlib_0[Data_Dictionary_streamlib_0['Doc_ID'] == doc_meta.split('|')[0].replace('Doc','')]['File_Code'].values[0]

    pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(file_name)
    card_2(
        'Doc{} '.format(doc_meta.split('|')[0]) + doc_name, 
        'Page{}'.format(doc_meta.split('|')[1]),
        '...{}...'.format(content),
        pdf_html,
    )

    # part_two_df = part_two_show_compare(df_dict_pair_2,Doc_Page_Sentence,Data_Dictionary_streamlib_2,doc_meta)
    part_two_df =part_two_show_compare(df_dict_pair_2,Doc_Page_Sentence_2,Data_Dictionary_streamlib_2,doc_meta)

    # st.dataframe(part_two_df)

    c21, c22 = st.columns((4, 4))
    with c21:
        for index in range(len(part_two_df)):
            doc_id = 'Doc' + part_two_df['Q_Doc_ID'].values[index]
            page_id = "Page" + part_two_df['Q_Page_ID'].values[index] + ' Sentence'  + part_two_df['Q_Sen_ID'].values[index]
            result_sentence = part_two_df['query_Sentence_show'].values[index]
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(Data_Dictionary_streamlib_0[Data_Dictionary_streamlib_0['Doc_ID'] == part_two_df['Q_Doc_ID'].values[index]]['File_Code'].values[0])
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
            pdf_html = """<a href="http://pc140032646.bot.or.th/th_pdf/{}" class="card-link">PDF</a>""".format(Data_Dictionary_streamlib_0[Data_Dictionary_streamlib_0['Doc_ID'] == part_two_df['R_Doc_ID'].values[index]]['File_Code'].values[0])
            card_4( 
                doc_id + ' ' + part_two_df['R_เรื่อง'].values[index],
                '{}'.format(conv.convert(result_sentence)),
                pdf_html,
                page_id,
            )