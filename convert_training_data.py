#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from genericpath import isfile
from os import listdir
import re
import codecs
import random
from requests.utils import quote
import requests
import time
GEONAMES_URL = "http://localhost:8091/search?query=GeonameId:"
def preprocess_FullHierarchy(text,record, max_hier=100):

    # Example text
    #text = "Atlanta, Fulton County (Fulton), Georgia (State of Georgia, GA, Peach State), United States (USA, U.S., United States of America, America, U.S.A., US), North America"
    
    # Define a regular expression pattern to match parentheses and their contents
    pattern = r'\((?:[^()]|(?R))*\)'
    
    # Use re.sub to replace the matched pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    # Split the cleaned text into a list of items using ', ' as the separator
    items = cleaned_text.split(', ')
    if len(items) > 1:
    # Remove the last item from the list
        items.pop()
    # import pdb
    # pdb.set_trace()
    items = [item.strip() for item in items]
    if len(items) > 1:
        if items[0] == items[1]:
            del items[1]
    if len(items) == 1 and 'Country' in record:
        items[0] = record['Country']
    #+äüü print(items)
    # Join the remaining items back into a single string with a space after the comma
    if len(items) > max_hier:
        items=items[0:max_hier]
    final_text = ', '.join(item.strip() for item in items)
    
    # Print the final cleaned text
    return final_text
    # print(final_text)


def insert_multiple_strings(original_string, strings_to_insert, insertion_indices):
    result = []
    previous_index = 0
    
    for index, insert_string in zip(insertion_indices, strings_to_insert):
        result.append(original_string[previous_index:index])
        result.append(insert_string)
        previous_index = index
    
    result.append(original_string[previous_index:])
    
    return ''.join(result)

data_list = ['lgl']        
hire_list = []
max_example_count = 50000
max_per_text = 100
total_c = 0
remove_count = 0
results = []
for test_data in data_list: #'lgl','neel', 'wiktor'
    io =  open(test_data+'.json',"r")
    true_dict = json.load(io)
    except_count = 0
    total_return_results = {}
    count = 0
    file_count = 0
    directory =  'data/'+test_data+ "/" 
    files = [f for f in listdir(directory) if isfile(directory + f)]
    for f in files:
        
        total_line = ''
        ID = f[0:len(f)-4]
        if ID not in true_dict:
            continue
        if not true_dict[ID]:
            continue
        exist_places = []
        # print(test_data, count,'#'*50, f)
        for line in codecs.open(directory + f, encoding="utf-8"):
            total_line += line
        # print(total_line)
        count = 0
        for place in true_dict[ID]:
            text = insert_multiple_strings(total_line, ['<START>','<END>'],[int(place['start']), int(place['end'])])
            # result = *evaluation_generator
            url = GEONAMES_URL+quote(place['geonamesID'].encode('utf8'))
            response = requests.get(url)
            jsondata = response.json()
            print(jsondata)
            
            if jsondata and len(jsondata["records"]) > 0:
                record = jsondata["records"][0]
                full_address = preprocess_FullHierarchy(record['FullHierarchy'],record,10)
                
                INSTRUCT = "Identify the full address of " + place['LOC'] + " (marked with <START> <END>) in the text. "
                results.append({'instruction': INSTRUCT, "input":text,"output":full_address})
                total_c +=1
                count += 1
            if count >= max_per_text:
                break
        if total_c > max_example_count:
            break


pattern = r'<START>(.*?)<END>'

random.shuffle(results)
json_file_path = "training_data.json"

# Open the file in write mode and save the data as JSON
with open(json_file_path, "w") as json_file:
    json.dump(results, json_file)
