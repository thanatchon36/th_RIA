# Network
from pyvis.network import Network
import matplotlib.pyplot as plt
# Search
import pythainlp
from pythainlp import sent_tokenize, word_tokenize
import numpy as np
import math
import pandas as pd
import re
import itertools
from pythainlp.corpus import thai_stopwords
import statistics
from colorama import Fore, Back, Style
from statistics import StatisticsError
import ast
# Compare
import pickle
import difflib as dif

class ria:
    def __init__(self):
        # File for Network
        self.df_dict_pair_0 = pd.read_csv('09_Output_Streamlib/df_dict_pair.csv')
        self.Data_Dictionary_streamlib_0 = pd.read_csv('09_Output_Streamlib/Data_Dictionary_streamlib.csv',dtype=str)

        # File for Search
        self.Doc_Page_Text_1 = pd.read_csv('09_Output_Streamlib/P_One_Doc_Page_Text.csv')
        self.category_text_1 = pd.read_csv('09_Output_Streamlib/category_text_score.csv')
        self.Data_Dictionary_streamlib_1 = pd.read_csv('09_Output_Streamlib/Data_Dictionary_streamlib.csv',dtype=str)
        self.df_dict_pair_1 = pd.read_csv('09_Output_Streamlib/df_dict_pair.csv')

        # File For Compare
        self.df_dict_pair_2 = pd.read_csv('09_Output_Streamlib/df_dict_pair.csv')
        self.Data_Dictionary_streamlib_2 = pd.read_csv('09_Output_Streamlib/Data_Dictionary_streamlib.csv',dtype=str)
        self.Doc_Page_Text_2 = pd.read_csv('09_Output_Streamlib/P_One_Doc_Page_Text.csv')
        self.Doc_Page_Sentence_2 = pd.read_csv('09_Output_Streamlib/Doc_Page_Sentence.csv')
        self.df_meta_question_answer = pd.read_csv('09_Output_Streamlib/df_meta_question_answer.csv')
        
        self.filter1_selected = list()
        self.filter2_selected = list()
        self.filter3_selected = list()
        
    #0
    def create_query_token_for_highlight(self,query):
        query_token = word_tokenize(query)
        stopwords = list(thai_stopwords())
        query_token_list = dict()
        query_token_stopwords = dict()
        stick_stopword = ''
        index = 0
        for i in query_token:
            if i not in stopwords:
                query_token_list[index] = i
                if len(stick_stopword) != 0:
                    query_token_stopwords[index] = stick_stopword
                    stick_stopword = ''
                index += 1
            else:
                stick_stopword += i
                #print(stick_stopword)

        return query_token_list,query_token_stopwords

    def creat_df_to_dict_for_highlight(self,df):
        df_dict = df.drop(columns=['score']).to_dict('records')
        new_dict = dict()
        for index_dict in df_dict:
            for key in index_dict.keys():
                new_dict[index_dict[key]] = key
        return new_dict

    def highlight_text(self, query_sentence, document):
        if all([True if or_token in query_sentence else False  for or_token in ['(','หรือ',')','"']]):
            query_sentence = query_sentence.replace('(','')
            query_sentence = query_sentence.replace('หรือ','')
            query_sentence = query_sentence.replace(')','')
            query_sentence = query_sentence.replace('"','')
            #if exact_match_use_other
        query_token_list,query_token_stopwords = self.create_query_token_for_highlight(query_sentence)
        df,candidate_df = self.find_min_location_token(document,query_token_list.values())
        df_dict = df.drop(columns=['score']).to_dict('records')
        new_dict = self.creat_df_to_dict_for_highlight(df)
        new_dict_keys_sorted = sorted(new_dict,reverse=True)

        for location in new_dict_keys_sorted:
            word_index_in_query_token_list = new_dict[location]
            len_token = len(query_token_list[word_index_in_query_token_list])
            replace_word_start = location-len_token
            replace_word_end = location
            if word_index_in_query_token_list in query_token_stopwords.keys():
                len_stop_word = len(query_token_stopwords[word_index_in_query_token_list])
                if document[replace_word_start-len_stop_word:replace_word_start] == query_token_stopwords[word_index_in_query_token_list]:
                    replace_word_start = replace_word_start-len_stop_word
            document = document[:replace_word_end] + '</mark>' + document[replace_word_end:]
            document = document[:replace_word_start] + f'<mark style="background-color:yellow;margin: 0;padding: 0;">' + document[replace_word_start:]
        return document
        
    def step3_2_click_show_result(self, query_sentence,compare_sentence):    
        query_sentence = self.create_query_token_for_compair(query_sentence.replace('\n',''))
        compare_sentence = self.create_query_token_for_compair(compare_sentence.replace('\n',''))
        compare_sentence_result_list = list(dif.Differ().compare(query_sentence,compare_sentence))
        
        new_str1 = ''
        new_str2 = ''
        len_first = 0 #เช็กว่าเป็นคำแรกของประโยคไหม ถ้าเป็นก็จะตัดออก เพื่อปรับให้ประโยคตรงกัน

        new_query_sentence = ''
        new_compare_sentence = ''
        '''update'''
        compare_sentence_result_list_start = self.trim_result_list(compare_sentence_result_list)
        compare_sentence_result_list_start.reverse()
        compare_sentence_result_list_end = self.trim_result_list(compare_sentence_result_list_start)
        compare_sentence_result_list_end.reverse()
        '''update end'''
        for symbol in compare_sentence_result_list_end:
            if symbol[0] == ' ':
                new_query_sentence += symbol[2:]
                new_compare_sentence += symbol[2:]
            elif symbol[0] == '-':
                new_query_sentence += f"{Fore.BLUE}{symbol[2:]}{Fore.BLACK}" # bleu
            elif symbol[0] == '+':
                new_compare_sentence += f"{Fore.RED}{symbol[2:]}{Fore.BLACK}" # Red

        return new_query_sentence.replace('BLANK',' '),new_compare_sentence.replace('BLANK',' ')
        
    #check if first words are differect, trim it
    def trim_result_list(self, compare_sentence_result_list):
        is_diff_at_start = True
        compare_sentence_result_list_trim = []
        for symbol in compare_sentence_result_list:
            if symbol[0] in ['-','+'] and is_diff_at_start:
                continue
            else:
                is_diff_at_start = False
                compare_sentence_result_list_trim.append(symbol)
        return compare_sentence_result_list_trim
        

    def part_two_show_compare(self, Doc_Page_ID):
        df_dict_pair = self.df_dict_pair_2
        Doc_Page_Sentence = self.Doc_Page_Sentence_2
        Data_Dictionary_streamlib = self.Data_Dictionary_streamlib_2
        
        Data_Dictionary_streamlib = Data_Dictionary_streamlib[['Doc_ID','เรื่อง']].copy()
        df_dict_pair[['Q_Doc_ID','Q_Page_ID','Q_Sen_ID']] = df_dict_pair['query'].str.split('|', expand=True)
        df_dict_pair[['R_Doc_ID','R_Page_ID','R_Sen_ID']] = df_dict_pair['result'].str.split('|', expand=True)
        df_dict_pair['Doc_Page_ID'] = df_dict_pair['Q_Doc_ID'] +'|'+df_dict_pair['Q_Page_ID']
        df_dict_pair_filter = df_dict_pair[df_dict_pair['Doc_Page_ID'] == Doc_Page_ID].copy()
        df_dict_pair_filter = df_dict_pair_filter.merge(Doc_Page_Sentence,right_on = 'Doc_Page_Sen_ID',left_on='query',how='left').drop(columns='Doc_Page_Sen_ID').rename(columns={'Sentence':'query_Sentence'})
        df_dict_pair_filter = df_dict_pair_filter.merge(Doc_Page_Sentence,right_on = 'Doc_Page_Sen_ID',left_on='result',how='left').drop(columns='Doc_Page_Sen_ID').rename(columns={'Sentence':'result_Sentence'})
        df_dict_pair_filter['All_Compare'] = df_dict_pair_filter.apply(lambda x: self.step3_2_click_show_result(x.query_Sentence, x.result_Sentence), axis=1)
        split_df = pd.DataFrame(df_dict_pair_filter['All_Compare'].tolist(), columns=['query_Sentence_show','result_Sentence_show'])
        result_all = pd.concat([df_dict_pair_filter,split_df], axis=1)
        result_all = result_all.drop(columns=['query_Sentence','result_Sentence','All_Compare']).sort_values(by='query')
        result_all = result_all.merge(Data_Dictionary_streamlib,right_on='Doc_ID',left_on='Q_Doc_ID',how='left')
        result_all = result_all.rename(columns={'เรื่อง':'Q_เรื่อง'}).drop(columns=['Doc_ID'])
        result_all = result_all.merge(Data_Dictionary_streamlib,right_on='Doc_ID',left_on='R_Doc_ID',how='left')
        result_all = result_all.rename(columns={'เรื่อง':'R_เรื่อง'}).drop(columns=['Doc_ID'])
        result_all.drop_duplicates(inplace=True)
        return result_all
        
    def part_one_show_original_text(self, Doc_Page_ID):
        Doc_Page_Text = self.Doc_Page_Text_2
        Data_Dictionary_streamlib = self.Data_Dictionary_streamlib_2
        
        Data_Dictionary_streamlib = Data_Dictionary_streamlib[['Doc_ID','เรื่อง']].copy()
        Doc_Page_Text[['Doc_ID','Page_ID']] = Doc_Page_Text['Doc_Page_ID'].str.split('|', expand=True)
        df_part_one = Doc_Page_Text[Doc_Page_Text['Doc_Page_ID'] == Doc_Page_ID].merge(Data_Dictionary_streamlib,on='Doc_ID',how='left')
        return df_part_one
        
    def create_query_token_for_compair(self, query):
        query_token = word_tokenize(query)
        return query_token
        
    def step1_user_search(self):
        query = self.query
        Doc_Page_Text = self.Doc_Page_Text_1
        category_text = self.category_text_1
        df_dict_pair = self.df_dict_pair_1
        Data_Dictionary_streamlib = self.Data_Dictionary_streamlib_1
        
        Doc_Page_Text['Score'] = Doc_Page_Text.apply(lambda x: self.retrieval_score(x['Original_text']), axis=1)
        Result_search = Doc_Page_Text.sort_values(by='Score', ascending=False)
        Result_search = Result_search[Result_search['Score'] > 0]
        Result_search['Score'] = Result_search['Score'].round(3)
        category_score = self.create_category_score(query)
        category_score = category_score.astype({'Category_Code': 'int'}).astype({'Category_Code': 'str'})
        Result_search[['Doc_ID', 'Page_ID']] = Result_search['Doc_Page_ID'].str.split('|', 1, expand=True)
        Result_search = Result_search.merge(Data_Dictionary_streamlib,on='Doc_ID',how='left')
        Result_search = Result_search.merge(category_score,on='Category_Code',how='left')
        Result_search = Result_search.sort_values(by=['Score','rank'],ascending=False)
        Result_search = Result_search.drop(columns=['File_Name','Cat_Score','rank'])
        df_dict_pair[['Q_Doc_ID','Q_Page_ID','Q_Sen_ID']] = df_dict_pair['query'].str.split('|', expand=True)
        df_dict_pair[['R_Doc_ID','R_Page_ID','R_Sen_ID']] = df_dict_pair['result'].str.split('|', expand=True)
        df_dict_pair['Doc_Page_ID'] = df_dict_pair['Q_Doc_ID'] + '|' + df_dict_pair['Q_Page_ID'] 
        df_dict_pair_filter_node = self.filter_node_for_search(df_dict_pair,Result_search).groupby('Doc_Page_ID')['result'].agg('count').reset_index().rename(columns={'result':'Number_result'})
        Result_search = Result_search.merge(df_dict_pair_filter_node,on='Doc_Page_ID',how='left')
        return Result_search
    '''fix 20220928'''
    def option_filter(self,Result_search):
        filter1_from_result = list(set([i for sublist in Result_search['สถาบันผู้เกี่ยวข้อง'] for i in ast.literal_eval(sublist)]))
        filter2_from_result = list(set([i for sublist in Result_search['ประเภทเอกสาร'] for i in ast.literal_eval(sublist)]))
        filter3_from_result = list(set([i for sublist in Result_search['กฎหมาย'] for i in ast.literal_eval(sublist)])) 
        return filter1_from_result, filter2_from_result, filter3_from_result

    def filter_col(self,row,row_names,selected_filter_lists):
        result_each_col = []
        y = []
        for row_number in range(len(row_names)):
            row_name = row_names[row_number]
            selected_filter_list = selected_filter_lists[row_number]
            if len(selected_filter_list) == 0:
                result_each_col.append(True)
            elif any([True for i in selected_filter_list if i in row[row_name]]):
                result_each_col.append(True)
            else:
                result_each_col.append(False)
        if all(result_each_col):
            return 1
        else:
            return 0

    def filter_result_search(self,Result_search):
        Result_search["Check"] = Result_search.apply(self.filter_col,
                                                     args=(['สถาบันผู้เกี่ยวข้อง','ประเภทเอกสาร','กฎหมาย'],
                                                           [self.filter1_selected,self.filter2_selected,self.filter3_selected]),
                                                     axis=1)
        Result_search = Result_search[Result_search['Check'] == 1].copy()
        Result_search = Result_search.drop(columns=['Check'])
        Result_search = Result_search.sort_values(by=['Score'], ascending=False).reset_index(drop=True)
        return Result_search
    '''end fix 20220928'''
    def filter_node_for_search(self, df_dict_pair,Result_search):
        Result_search_unique = Result_search['Doc_Page_ID'].unique()
        df_dict_pair_filter = df_dict_pair[df_dict_pair['Doc_Page_ID'].isin(Result_search_unique)]
        return df_dict_pair_filter
        
    def create_category_score(self, query):
        category_text = self.category_text_1
        query_list = self.create_query_list(query)
        all_query_token = list(set([token for token in self.create_query_token(query) for query in query_list]))
        try:
            all_query_token.remove('หรือ')
            all_query_token.remove('(')
            all_query_token.remove(')')
        except ValueError:
            pass
        filter_col = list(filter(lambda col: col in query_list , category_text.columns[1:]))
        filter_col.append('cat')
        df_score = pd.DataFrame(data = {'Category_Code':category_text['cat'],'Cat_Score':category_text[filter_col].sum(axis = 1)})
        df_score = df_score.sort_values(by='Cat_Score',ascending=False)
        df_score['rank'] = [str(i) for i in range(len(df_score)-1,-1,-1)]
        df_score = df_score.reset_index(drop=True)
        return df_score
        
    def find_min_location_token(self, document,query_token):
        token_location_all = self.find_token_location_in_doc(document,query_token)
        if len(query_token) > 1:
            df = self.find_candidate_df(token_location_all)
            candidate_df = self.find_candidate_df(token_location_all)
            for column in df.columns[:-1]:
                df_group = df.groupby(by=[column])['score'].agg('min').reset_index()
                df = df.merge(df_group,how='inner', on=[column,'score'])
        else:
            df = self.find_candidate_df_for_len_one(token_location_all)
            candidate_df = self.find_candidate_df_for_len_one(token_location_all)
        return df,candidate_df
        
    def retrieval_score(self, document):
        query = self.query
        #update 20221006 exect_match
        if '"' in query:
            query = query.replace('"','')
            retrieval = len(list(re.finditer(query, document)))
            return retrieval
        query_list = self.create_query_list(query)
        df = pd.DataFrame()
        candidate_df = pd.DataFrame()
        for query in query_list:
            query_token = self.create_query_token(query)
            query_token = list(filter(lambda token: token != ' ', query_token))
            try:
                df_,candidate_df = self.find_min_location_token(document,query_token)
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
        
    def find_candidate_df_for_len_one(self, token_location_all):
        df = pd.DataFrame(data={0:token_location_all[0],'score':[1 for i in range(0,len(token_location_all[0]))]})
        return df
        
    #ไม่สามารถค้นหาคำเดียวได้จึงต้องใช้find_candidate_df_for_len_one
    def find_candidate_df(self, token_location_all):
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
        
    def find_token_location_in_doc(self, document,query_token):
        token_location_all = []
        for token in query_token:
            location_token_list = []
            for location_token in re.finditer(token.upper(), document.upper()):
                 location_token_list.append(location_token.span()[1])
            token_location_all.append(location_token_list)
        return token_location_all
        
    def create_query_token(self, query):
        query_token = word_tokenize(query)
        stopwords = list(thai_stopwords())
        query_token = [i for i in query_token if i not in stopwords]
        return query_token
        
    def create_query_list(self, query):
        open_bracket_location = []
        close_bracket_location = []
        for bracket in re.finditer('\(', query):
            open_bracket_location.append(bracket.span()[1])
        for bracket in re.finditer('\)', query):
            close_bracket_location.append(bracket.span()[0])
        pair_bracket_location = []
        if len(open_bracket_location) == len(close_bracket_location):
            for index in range(len(open_bracket_location)):
                pair_bracket_location.append((open_bracket_location[index],close_bracket_location[index]))

        query_split_all = []
        for pair in pair_bracket_location:
            query_splits = query[pair[0]:pair[1]].split('หรือ')
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
        
    def filter_node(self, df_dict_pair,Result_search):
        Result_search_unique = Result_search['Doc_Page_ID'].unique()
        df_dict_pair_filter = df_dict_pair[df_dict_pair['Doc_Page_ID'].isin(Result_search_unique)]
        df_dict_pair_filter_no_pair = pd.DataFrame(data={'Doc_Page_ID':list(set(Result_search_unique).difference(set(df_dict_pair['Doc_Page_ID'])))})
        df_dict_pair_filter_no_pair[['Q_Doc_ID','Q_Page_ID']] = df_dict_pair_filter_no_pair['Doc_Page_ID'].str.split('|', expand=True)
        return df_dict_pair_filter,df_dict_pair_filter_no_pair

    #500 * 2800
    def create_network(self, Result_search):
        df_dict_pair = self.df_dict_pair_0
        Data_Dictionary_streamlib = self.Data_Dictionary_streamlib_0        
        df_dict_pair[['Q_Doc_ID','Q_Page_ID','Q_Sen_ID']] = df_dict_pair['query'].str.split('|', expand=True)
        df_dict_pair[['R_Doc_ID','R_Page_ID','R_Sen_ID']] = df_dict_pair['result'].str.split('|', expand=True)
        df_dict_pair['Doc_Page_ID'] = df_dict_pair['Q_Doc_ID'] + '|' + df_dict_pair['Q_Page_ID'] 
        df_dict_pair_filter ,df_dict_pair_filter_no_pair = self.filter_node(df_dict_pair,Result_search)

        all_pair_Doc_id = df_dict_pair_filter[['Q_Doc_ID','R_Doc_ID']].copy()
        all_pair_Doc_id['Count'] = 1
        all_pair_Doc_id_group = all_pair_Doc_id.groupby(['Q_Doc_ID','R_Doc_ID'])['Count'].agg('count').reset_index()
        #print(all_pair_Doc_id_group)
        median_score = statistics.median(all_pair_Doc_id_group['Count'])
        all_node = list(all_pair_Doc_id_group['Q_Doc_ID'].unique())
        all_node.extend(all_pair_Doc_id_group['R_Doc_ID'].unique())
        all_node = list(set(all_node))
        G = Network(height='500px', width='100%',bgcolor="#f2f2f2")  #222222

        for Doc_ID in df_dict_pair_filter_no_pair['Q_Doc_ID'].unique():
            thai_name_doc = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เลขที่ (Thai)'].iloc[0]
            Doc_Name = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เรื่อง'].iloc[0]
            G.add_node(Doc_ID,label=thai_name_doc,title=["ประกาศหลัก:"+'\n'+thai_name_doc+' :'+Doc_Name],shape='dot') #circle

        for Doc_ID in all_node:
            Doc_Name = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เรื่อง'].iloc[0]
    #         Doc_ID_Name_len = len(Doc_Name)
    #         if Doc_ID_Name_len > 100:
    #             Doc_Name = Doc_Name[:round(Doc_ID_Name_len/2)]+'\n'+Doc_Name[round(Doc_ID_Name_len/2):]
            thai_name_doc = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เลขที่ (Thai)'].iloc[0]
            G.add_node(Doc_ID,label=thai_name_doc,title=["ประกาศหลัก:"+'\n'+thai_name_doc+' :'+Doc_Name],shape='dot')
        try:
            for Q_Doc_ID in all_pair_Doc_id_group['Q_Doc_ID'].unique():
                Number_connect_nodes = len(all_pair_Doc_id_group[all_pair_Doc_id_group['Q_Doc_ID'] == Q_Doc_ID])
                for Number_connect_node in range(Number_connect_nodes):
                    R_Doc_ID = all_pair_Doc_id_group[all_pair_Doc_id_group['Q_Doc_ID'] == Q_Doc_ID]['R_Doc_ID'].iloc[Number_connect_node]
                    weight = all_pair_Doc_id_group[all_pair_Doc_id_group['Q_Doc_ID'] == Q_Doc_ID]['Count'].iloc[Number_connect_node]
                    if weight > median_score:
                        value = 64
                    else:
                        value = 56
                    G.add_edge(str(Q_Doc_ID), str(R_Doc_ID), value=str(value),title='จำนวนคู่ที่เหมือนกัน:'+str(weight))
            neighbor_map = G.get_adj_list()               
            for node in G.nodes:
                if len(neighbor_map[node['id']]) >=1:
                    node['title'][0] += '\n\n ประกาศที่เกี่ยวข้อง:\n'
                #print(node,node['size'])
                for Doc_ID in neighbor_map[node['id']]:
                    thai_name_doc = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เลขที่ (Thai)'].iloc[0]
                    Doc_ID_Name = Data_Dictionary_streamlib[Data_Dictionary_streamlib['Doc_ID'] == Doc_ID]['เรื่อง'].iloc[0]
                    node['title'][0] += f' {thai_name_doc} :'+Doc_ID_Name+ '\n'
        except:
            pass
    #     "border": "rgba(34, 42, 89,1)",
    #     "background": "rgba(11, 81, 89,1)",
        G.set_options('''
    var options = {
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 1,
        "font": {
          "color": "rgba(114, 114, 115,1)",
          "size": 25
          },
        "color": {
          "border": "rgba(123, 149, 166,1)",
          "background": "rgba(22, 79, 115,1)",
          "highlight": {
            "border": "rgba(22, 79, 115,1)",
            "background": "rgba(107, 204, 242,1)"
          },
          "hover": {
            "border": "rgba(22, 79, 115,1)",
            "background": "rgba(107, 204, 242,1)"
          }
        }
      },
      "edges": {
        "color": {
          "color": "rgba(114, 114, 115,1)",
          "hover": "rgba(142, 191, 107,1)",
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
          "centralGravity": 1.75,
          "springLength": 45,
          "springConstant": 0.001
        },
        "minVelocity": 0.75
      }
    }
    ''')
        return G        
