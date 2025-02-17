import os
# from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader,  TextLoader
import json



def get_document_text(json_file_path, dict_file_path):
    with open(json_file_path, 'r') as file:
        data_str = file.read()
    data = json.loads(data_str)

    res_dict = dict()
    for item in data:
        if item['id'] not in res_dict:
            res_dict[item['id']] = item
        else:
            print (item['id'])

    loader = DirectoryLoader(dict_file_path, glob="**/*.txt", loader_cls=TextLoader)
    data = loader.load()
    # print(data)

    # Split the text into Chunks
    if data[0].metadata['source'][:13] == '../text_files':
        for item in data:
            item.metadata['date'] = res_dict[item.metadata['source'][14:-4]]['date'][:10]
            item.metadata['title'] = res_dict[item.metadata['source'][14:-4]]['collection_title']
            item.metadata['title_en'] = res_dict[item.metadata['source'][14:-4]]['title_en']
            item.metadata['ancestor_title'] = res_dict[item.metadata['source'][14:-4]]['ancestor_titles']
            item.metadata['full_text'] = res_dict[item.metadata['source'][14:-4]]['full_text']
    else:
        for item in data:
            item.metadata['date'] = res_dict[item.metadata['source'][21:-4]]['date'][:10]
            item.metadata['title'] = res_dict[item.metadata['source'][21:-4]]['collection_title']
            item.metadata['title_en'] = res_dict[item.metadata['source'][21:-4]]['title_en']
            item.metadata['ancestor_title'] = res_dict[item.metadata['source'][21:-4]]['ancestor_titles']
            item.metadata['full_text'] = res_dict[item.metadata['source'][21:-4]]['full_text']

        # ancestor_titles": [
        #      "The Glamorgan Monmouth and Brecon Gazette and Merthyr Guardian",
        #     "[4]"
        # ],s


    return data
