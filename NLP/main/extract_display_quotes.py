#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:55:46 2022

@author: sjuf9909
"""

# import required packages
import os
import io
import sys
import codecs
import utils
import logging
import traceback
from collections import Counter

# matplotlib: visualization tool
from matplotlib import pyplot as plt

# pandas: tools for data processing
import pandas as pd

# spaCy and NLTK: natural language processing tools for working with language/text data
import spacy
from spacy import displacy
from spacy.tokens import Span
import nltk
nltk.download('punkt')
from nltk import Tree
from nltk.tokenize import sent_tokenize

# ipywidgets: tools for interactive browser controls in Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython.display import display, Markdown, clear_output

# import the quote extractor tool
# Source: https://github.com/sfu-discourse-lab/GenderGapTracker
from quote_extractor import extract_quotes, get_rawtext_files
from config import config


class QuotationTool():
    '''
    Interactive tool for extracting and displaying quotes in a text
    '''
    
    def __init__(self):
        '''
        Initiate the QuotationTool
        '''
        # initiate the app_logger
        self.app_logger = utils.create_logger('quote_extractor', log_dir='logs', 
                                         logger_level=logging.INFO,
                                         file_log_level=logging.INFO)
        
        # download spaCy's en_core_web_lg, the pre-trained English language tool from spaCy
        print('Loading spaCy language model...')
        self.nlp = spacy.load('en_core_web_lg')
        print('Finished loading.')
        
    
    def upload_files(self, file_type):
        '''
        Upload text or excel files as input to the QuotationTool
        
        Args:
            file_type: the file type being uploaded ('text' for uploading .txt files
                                                     or 'excel' for uploading .xlsx file)
        '''
        if file_type=='text':
            print('Upload your .txt files here:')
            # widget to upload .txt files
            uploader = widgets.FileUpload(
                accept='.txt', # accepted file extension 
                multiple=True  # True to accept multiple files
                )
        elif file_type=='excel':
            print('Upload your excel spreadsheet (.xlsx) here:')
            # widget to upload .xlsx file
            uploader = widgets.FileUpload(
                accept='.xlsx', # accepted file extension
                multiple=False  # to accept one Excel file only
                )
        else:
            # warning if other type files are uploaded
            print('You can only upload .txt or .xlsx files!')
        
        # give notification when file is uploaded
        def _cb(change):
            clear_output()
            print('File uploaded!')
            
        uploader.observe(_cb, names='data')
        
        return uploader
    
    
    def nlp_preprocess(self, text):
        '''
        Pre-process and create spaCy text

        Args:
            text: the text to be processed
        '''
        # pre-process text
        text = sent_tokenize(text)
        text = ' '.join(text)
        text = utils.preprocess_text(text)
        
        # apply the spaCy's tool to the text
        doc = self.nlp(text)
        
        return doc
    
    
    def process_txt(self, txt_upload):    
        '''
        Pre-process uploaded .txt files into pandas dataframe

        Args:
            txt_upload: the uploaded .txt files from upload_files()
        '''
        # create an empty list for a placeholder to store all the texts
        all_files = []
        
        # search for text files (.txt) inside the folder and extract all the texts
        for input_file in txt_upload.value.keys():
            text_dict = {}
            
            # use the text file name as the doc_id
            doc_id = input_file.replace('.txt', '')
            
            try:
                # read the text file
                doc_lines = codecs.decode(txt_upload.value[input_file]['content'], encoding='utf-8')
                
                # store them inside a dictionary
                text_dict['text_id'] = doc_id
                text_dict['text'] = doc_lines
                all_files.append(text_dict)
                    
            except:
                # this will provide some information in the case of an error
                app_logger.exception("message")
                traceback.print_exc()
        
        # convert the extracted texts into a pandas dataframe for further processing
        text_df = pd.DataFrame.from_dict(all_files)
        text_df['spacy_text'] = text_df['text'].apply(lambda text: self.nlp_preprocess(text))
        text_df.set_index('text_id', inplace=True)
        
        return text_df
    
    
    def process_xls(self, xls_upload):
        '''
        Pre-process uploaded .xlsx file into pandas dataframe

        Args:
            xls_upload: the uploaded .xlsx file from upload_files()
        '''
        # read the excel file containing the list of texts and convert them into a pandas dataframe
        text_df = pd.read_excel(io.BytesIO(xls_upload.data[0]))
        text_df['spacy_text'] = text_df['text'].apply(lambda text: self.nlp_preprocess(text))
        text_df.set_index('text_id', inplace=True)
        
        return text_df
        

    def get_quotes(self, text_df, inc_ent, create_tree=False):
        '''
        Extract quotes and their meta-data (quote_id, quote_index, etc.) from the text
        and return as a pandas dataframe

        Args:
            text_df: the pandas dataframe containing the list of texts
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
            create_tree: option to create parse tree files for the quotes 
        '''
        # create an output folder and specify the file path if create_tree=True
        if create_tree:
            os.makedirs('output', exist_ok=True)
            os.makedirs('output/trees', exist_ok=True)
            tree_dir = './output/trees/'
        else:
            tree_dir = None
    
        # create an empty list to store all detected quotes
        all_quotes = []
        
        # go through all the texts and start extracting quotes
        for row in text_df.itertuples():
            doc_id = row.Index
            doc = row.spacy_text
            
            try:        
                # extract the quotes
                quotes = extract_quotes(doc_id=doc_id, doc=doc, 
                                        write_tree=create_tree, 
                                        tree_dir=tree_dir)
                
                # extract the included named entities
                speaks, qts = [quote['speaker'] for quote in quotes], [quote['quote'] for quote in quotes]        
                speak_ents = [[(str(ent), ent.label_) for ent in doc.ents if (str(ent) in speak) & (ent.label_ in inc_ent)] for speak in speaks]        
                quote_ents = [[(str(ent), ent.label_) for ent in doc.ents if (str(ent) in qt) & (ent.label_ in inc_ent)] for qt in qts]
        
                # add text_id, quote_id and named entities to each quote
                for n, quote in enumerate(quotes):
                    quote['text_id'] = doc_id
                    quote['quote_id'] = str(n)
                    quote['speaker_entities'] = list(set(speak_ents[n]))
                    quote['quote_entities'] = list(set(quote_ents[n]))
                    
                # store them in all_quotes
                all_quotes.extend(quotes)
                    
            except:
                # this will provide some information in the case of an error
                self.app_logger.exception("message")
                traceback.print_exception()
                
        # convert the outcome into a pandas dataframe
        quotes_df = pd.DataFrame.from_dict(all_quotes)
        
        # convert the string format quote spans in the index columns to a tuple of integers
        for column in quotes_df.columns:
            if column.endswith('_index'):
                quotes_df[column].replace('','(0,0)', inplace=True)
                quotes_df[column] = quotes_df[column].apply(eval)
        
        # re-arrange the columns
        new_index = ['text_id', 'quote_id', 'quote', 'quote_index', 'quote_entities', 
                     'speaker', 'speaker_index', 'speaker_entities',
                     'verb', 'verb_index', 'quote_token_count', 'quote_type', 'is_floating_quote']
        quotes_df = quotes_df.reindex(columns=new_index)
                
        return quotes_df
    
    
    def show_quotes(self, text_df, quotes_df, text_id, show_what, inc_ent, save_to_html=False):
        '''
        Display speakers, quotes and named entities inside the text using displaCy

        Args:
            text_df: the pandas dataframe containing the list of texts
            quotes_df: the pandas dataframe containing the quotes
            text_id: the text_id of the text you wish to display
            show_what: options to display speakers, quotes or named entities
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
            save_to_html: option to save the display into an html file
        '''
        # formatting options
        TPL_SPAN = '''
        <span style="font-weight: bold; display: inline-block; position: relative; 
        line-height: 55px">
            {text}
            {span_slices}
            {span_starts}
        </span>
        '''
        
        TPL_SPAN_SLICE = '''
        <span style="background: {bg}; top: {top_offset}px; height: 4px; left: -1px; width: calc(100% + 2px); position: absolute;">
        </span>
        '''
        
        TPL_SPAN_START = '''
        <span style="background: {bg}; top: {top_offset}px; height: 4px; border-top-left-radius: 3px; border-bottom-left-radius: 3px; left: -1px; width: calc(100% + 2px); position: absolute;">
            <span style="background: {bg}; z-index: 10; color: #000; top: -0.5em; padding: 2px 3px; position: absolute; font-size: 0.6em; font-weight: bold; line-height: 1; border-radius: 3px">
                {label}{kb_link}
            </span>
        </span>
        '''
        
        colors = {'QUOTE': '#66ccff', 'SPEAKER': '#66ff99'}
        options = {'ents': ['QUOTE', 'SPEAKER'], 
                   'colors': colors, 
                   'top_offset': 42,
                   'template': {'span':TPL_SPAN,
                               'slice':TPL_SPAN_SLICE,
                               'start':TPL_SPAN_START},
                   'span_label_offset': 14,
                   'top_offset_step':14}
        
        # get the spaCy text 
        doc = text_df.loc[text_id, 'spacy_text']
        
        # create a mapping dataframe between the character index and token index from the spacy text.
        loc2tok_df = pd.DataFrame([(t.idx, t.i) for t in doc], columns = ['loc', 'token'])
    
        # get the quotes and speakers indexes
        locs = {
            'QUOTE': quotes_df[quotes_df['text_id']==text_id]['quote_index'].tolist(),
            'SPEAKER': set(quotes_df[quotes_df['text_id']==text_id]['speaker_index'].tolist())
        }
    
        # create the displaCy code to visualise quotes and speakers
        my_code_list = ['doc.spans["sc"] = [', ']']
        
        for key in locs.keys():
            for loc in locs[key]:
                if loc!=(0,0):
                    # Find out all token indices that falls within the given span (variable loc)
                    selTokens = loc2tok_df.loc[(loc[0]<=loc2tok_df['loc']) & (loc2tok_df['loc']<loc[1]), 'token'].tolist()
                    
                    # option to display named entities only
                    if show_what==['NAMED ENTITIES']:
                        for ent in doc.ents:
                            if (ent.start in selTokens) & (ent.label_ in inc_ent):
                                span_code = "Span(doc, {}, {}, '{}'),".format(ent.start, 
                                                                  ent.end, 
                                                                  ent.label_) 
                                my_code_list.insert(1,span_code)
                    
                    # option to display speaker and/or quotes and/or named entities
                    elif key in show_what:
                        if 'NAMED ENTITIES' in show_what:
                            for ent in doc.ents:
                                if (ent.start in selTokens) & (ent.label_ in inc_ent):
                                    span_code = "Span(doc, {}, {}, '{}'),".format(ent.start, 
                                                                      ent.end, 
                                                                      ent.label_) 
                                    my_code_list.insert(1,span_code)
                        
                        start_token, end_token = selTokens[0], selTokens[-1] 
                        span_code = "Span(doc, {}, {}, '{}'),".format(start_token, end_token+1, key) 
                        my_code_list.insert(1,span_code)
                    
        # combine all codes
        my_code = ''.join(my_code_list)
    
        # execute the code
        exec(my_code)
        
        # display the preview in this notebook
        displacy.render(doc, style='span', options=options, jupyter=True)
    
        # option to save the preview as an html document
        if save_to_html:
            # create an output folder if not yet available
            os.makedirs('output', exist_ok=True)
            out_dir='./output/'
            
            # render the html preview
            html = displacy.render(doc, style='span', options=options, jupyter=False, page=True)
            
            # save the preview as an html file
            file = open(out_dir+text_id+'.html', 'w')
            file.write(html)
            file.close()
        
        
    def top_entities(self, quotes_df, text_id, which_ent='speaker_entities',top_n=5):
        '''
        Display top n named entities inside the text using displaCy

        Args:
            quotes_df: the pandas dataframe containing the quotes
            text_id: the text_id of the text you wish to display
            which_ent: option to display named entities in speakers ('speaker_entities') 
                       or quotes ('quote_entities')
            top_n: the number of entities to display
        '''
        # get the top entities
        most_ent = quotes_df[quotes_df['text_id']==text_id][which_ent].tolist()
        most_ent = list(filter(None,most_ent))
        most_ent = [ent for most in most_ent for ent in most]
        most_ent = Counter([ent_name for ent_name, ent_label in most_ent])
        top_ent = dict(most_ent.most_common()[:top_n])
        
        # visualize the top entities
        bar_colors = {'speaker_entities':'#2eb82e',
                      'quote_entities':'#008ae6'}
        plt.figure(figsize=(10, 2.5))
        plt.bar(top_ent.keys(), top_ent.values(), color=bar_colors[which_ent])
        plt.yticks(range(0, most_ent[max(most_ent, key=most_ent.get)]+1, 1))
        plt.title('Top {} {} in {}'.format(min(top_n,len(top_ent.keys())),which_ent,text_id))
        plt.show()
        

    def analyse_quotes(self, text_df, quotes_df, inc_ent):
        '''
        Interactive tool to display and analyse speakers, quotes and named entities inside the text

        Args:
            text_df: the pandas dataframe containing the list of texts
            quotes_df: the pandas dataframe containing the quotes
            inc_ent: a list containing the named entities to be extracted from the text, 
                     e.g., ['ORG','PERSON','GPE','NORP','FAC','LOC']
        '''
        # widget for entering text_id
        enter_text = widgets.HTML(
            value="<b>Enter text_id:</b>",
            placeholder='',
            description=''
            )
        
        text = widgets.Text(
            value='',
            description='',
            style=dict(font_style='italic', fontweight='bold'),
            layout = widgets.Layout(width='100px')
            )
        
        # widgets to select what to preview, i.e., speaker and/or quote and/or named entities
        entity_options = widgets.HTML(
            value="<b>Select what entity to show:</b>",
            placeholder='',
            description='',
            )
        
        speaker_box = widgets.Checkbox(
            value=False,
            description='Speaker',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        quote_box = widgets.Checkbox(
            value=False,
            description='Quote',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        ne_box = widgets.Checkbox(
            value=False,
            description='Named Entities',
            disabled=False,
            indent=False,
            layout=Layout(margin='0px 0px 0px 0px')
            )
        
        # widget to show the preview
        preview_button = widgets.Button(description='Click to preview', 
                                        layout=Layout(margin='10px 0px 0px 10px'),
                                        style=dict(font_style='italic',
                                                   font_weight='bold'))
        preview_out = widgets.Output()
        
        def on_preview_button_clicked(_):
            with top_out:
                clear_output()
            with preview_out:                                                                                   
                # what happens when we click the preview_button
                clear_output()
                text_id = text.value
                show_what = []
                if speaker_box.value:
                    show_what.append('SPEAKER')
                if quote_box.value:
                    show_what.append('QUOTE')
                if ne_box.value:
                    show_what.append('NAMED ENTITIES')
                if show_what==[]:
                    print('Please select the entities to display!')
                else:
                    try:
                        self.show_quotes(text_df, quotes_df, text_id, show_what, inc_ent, save_to_html=False)
                    except:
                        print('Please enter the correct text_id!')
        
        # link the preview_button with the function
        show_what = preview_button.on_click(on_preview_button_clicked)
        
        # widget to save the preview
        save_button = widgets.Button(description='Save preview', 
                                     layout=Layout(margin='10px 0px 0px 10px'),
                                     style=dict(font_style='italic',
                                                font_weight='bold'))
        
        def on_save_button_clicked(_):
            with preview_out:
                # what happens when we click the save_button
                clear_output()
                text_id = text.value
                show_what = []
                if speaker_box.value:
                    show_what.append('SPEAKER')
                if quote_box.value:
                    show_what.append('QUOTE')
                if ne_box.value:
                    show_what.append('NAMED ENTITIES')
                if show_what==[]:
                    print('Please select the entities to display!')
                else:
                    try:
                        self.show_quotes(text_df, quotes_df, text_id, show_what, inc_ent, save_to_html=True)
                        print('Preview saved!')
                    except:
                        print('Please enter the correct text_id!')
        
        # link the save_button with the function
        save_button.on_click(on_save_button_clicked)
        
        # widget to show top 5 entities
        top_button = widgets.Button(description='Top 5 entities', 
                                     layout=Layout(margin='10px 0px 0px 10px'),
                                     style=dict(font_style='italic',
                                                font_weight='bold'))
        top_out = widgets.Output()
        
        def on_top_button_clicked(_):
            with top_out:
                # what happens when we click the top_button
                clear_output()
                text_id = text.value
                try:
                    self.top_entities(quotes_df, text_id, which_ent='speaker_entities',top_n=5)
                    self.top_entities(quotes_df, text_id, which_ent='quote_entities',top_n=5)
                except:
                    print('Please ensure you have entered the correct text_id and select the entities to display!')
        
        # link the top_button with the function
        top_button.on_click(on_top_button_clicked)
        
        # displaying buttons and their outputs
        vbox1 = widgets.VBox([enter_text, text, entity_options, speaker_box, quote_box, ne_box,
                              preview_button, save_button, top_button])
        #vbox2 = widgets.VBox([preview_button, save_button, top_button])
        
        hbox = widgets.HBox([vbox1, top_out])
        vbox = widgets.VBox([hbox, preview_out])
        return vbox