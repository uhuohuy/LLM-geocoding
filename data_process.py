#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:28:01 2024

@author: hu_xk
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
from genericpath import isfile
from os import listdir
import json


'''This is to cut down the long text and keep the targt toponym in the center of the shorter text'''
def find_centered_substring(text, max_chars=1000):
    # Find the position of <START> and <END>
    start_idx = text.find('<START>')
    end_idx = text.find('<END>')

    if start_idx != -1 and end_idx != -1:
        # Calculate the start and end character indices for the centered substring
        start_char_idx = max(start_idx - max_chars, 0)
        end_char_idx = min(end_idx + max_chars, len(text))

        # Find the nearest whitespace to the left of the start_char_idx
        while start_char_idx > 0 and text[start_char_idx] != ' ':
            start_char_idx -= 1

        # Find the nearest whitespace to the right of the end_char_idx
        while end_char_idx < len(text) - 1 and text[end_char_idx] != ' ':
            end_char_idx += 1

        # Extract the centered substring based on character indices
        centered_substring = text[start_char_idx:end_char_idx + 1]

        return centered_substring

    return ""

def insert_multiple_strings(original_string, strings_to_insert, insertion_indices):
    result = []
    previous_index = 0
    
    for index, insert_string in zip(insertion_indices, strings_to_insert):
        result.append(original_string[previous_index:index])
        result.append(insert_string)
        previous_index = index
    
    result.append(original_string[previous_index:])
    
    return ''.join(result)

max_char = 1000
data_list = ['geocorpora','19th','wotr','trnews','gwn','LDC','wiktor']       # ,'geovirus','TUD','neel','semeval'
for test_data in data_list: 
    io =  open('data/'+test_data+'.json',"r")
    true_dict = json.load(io)

    count = 0

    directory =  'data/'+ test_data+ "/" 
    files = [f for f in listdir(directory) if isfile(directory + f)]
    for f in files:
        count += 1            
        total_line = ''
        ID = f[0:len(f)-4]
        if ID not in true_dict:
            continue
        if not true_dict[ID]:
            continue
        exist_places = []
        print(test_data, count,'#'*50, f)
        for line in codecs.open(directory + f, encoding="utf-8"):
            total_line += line
        # print(total_line)
        return_objects = []
        for place in true_dict[ID]:
            text = insert_multiple_strings(total_line, ['<START>','<END>'],[int(place['start']), int(place['end'])])
            text = find_centered_substring(text, max_char)
            print(text)
