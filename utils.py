import os 
import re
from collections import Counter
from unstructured.partition.pdf import partition_pdf
import tiktoken
from weaviate.gql.get import HybridFusion
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from time import sleep



PROTECTED_CLASSES = ['Morningstar_json_chunking_test_quant','Articles', 'ArticlesTest', 'Articles_v1', 'Articles_v2', 'Chunking_quant', 'DOC_STATUS_V1', 'Datapoints', 'DirectDataPointPOC01', 'DirectDataPointPOC02', 'DirectDataPointPOC03', 'DirectDataPointPOCNB01', 'DirectDocs', 'DirectPOC', 'DirectPOC01', 'DirectUniversePOC02', 'Document', 'EQUITY_V1', 'FundStockData', 'GPTCache', 'GitBookChunk', 'HackathonCompanyInfo', 'Investments', 'Legal_filings', 'MstarAnalyticslabChatbot', 'MstarAnalyticslabChatbot_v1', 'MstarCompassSupport', 'MstarDatapoints', 'MstarDirectLens', 'MstarDirectPs', 'MstarInvestments', 'MstarRetirementRmChatbot', 'MstarRetirementRmChatbot_Ignacio', 'MstarRetirementRmChatbot_v1', 'MstarSalesforceArticles', 'MstarSalesforceArticles_v1', 'MstarWealthBaa', 'MstarWorkplaceAtlas', 'MstarWorkplaceRmChatbot', 'MstarWorkplaceRmChatbot_v1', 'MstarWsSupport', 'MstarWsSupport_v1', 'Pen_test', 'QuestionClassifier', 'QuestionClassifierTest1', 'QuestionClassifierTest_old_data', 'QuestionClassifier_v2', 'Questions', 'Test_chunk_ingestion', 'Test_chunk_ingestion_1', 'Test_ingest', 'Test_ingest_a', 'TestClass', 'IngestTest', 'IngestTest1', 'IngestTest2', 'MOBOT461', 'Testendpoint', 'Testendpointwoproperties', 'Testendpointproperties', 'Testendpointpassingproperties', 'TestCorpusCreation', 'TestCorpusCreate', 'TestCorpusCreateB']

def parse_pdf_w_unstructured(file_path, include_page_breaks):
    
    elements_uns = partition_pdf(file_path, languages=["eng"], strategy='fast', include_page_breaks=include_page_breaks)

    # Assuming elements_fast is a list of objects that can be converted to dictionaries
    elements = [el.to_dict() for el in elements_uns]

    # Keys you want to keep
    keys_to_keep = ['type', 'text','metadata']

    # List to hold filtered dictionaries
    filtered_elements_fast = []

    # Loop through each dictionary in the list
    for ele in elements:
        # Create an empty dictionary for the filtered elements
        filtered_dict_fast = {}
        # Loop through each key-value pair in the current dictionary
        for k, v in ele.items():
            # If the key is one of the ones we want to keep, add it to the filtered dictionary
            if k in keys_to_keep:
                filtered_dict_fast[k] = v.strip() if type(v) ==str else v
        # After filtering the dictionary, append it to the list of filtered dictionaries
        filtered_elements_fast.append(filtered_dict_fast)
        
    return filtered_elements_fast, elements_uns



def get_number_of_tokens(string):
    enc = tiktoken.get_encoding("cl100k_base")

    encoded = enc.encode(string)
    return len(encoded)

def num_tokens_from_string(string: str, encoding_name= "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def split_by_page(elements): #uns_elemets
    pages_text = []
    text = ''
    page_number = 1  # Initialize page number if not included in the first element
    for ele in elements:
        if ele['type'] == 'PageBreak':
            # Append the accumulated text to the list with the current page number
            pages_text.append({'token_count': num_tokens_from_string(text.strip()),'page_text': text.strip(), 'page_number': page_number})
            text = ''  # Reset the text for the next page
            page_number += 1  # Increment the page number for the next page
        else:
            text += " " + ele['text']  # Accumulate text for the current page

    # After the loop, check if there is any text left to be added to the last page
    if text:
        pages_text.append({'token_counr': num_tokens_from_string(text.strip()),'page_text': text.strip(), 'page_number': page_number})

    return pages_text

def create_lg_docs(pages):
    docs=[]
    for page in pages:
        # Create a Document object with page content from the chunk and the provided metadata.
        doc = Document(page_content=page['page_text'], metadata={'page_number':str(page['page_number'])})
        docs.append(doc)

    return docs

def page_chunking(elements):
    pages=split_by_page(elements)
    update_pages = []

    for idx, page in enumerate(pages):
        if idx == 0:
            # Directly append the first page as there is no previous page to prepend from
            update_pages.append(page)
        else:
            # Fetch current and previous pages
            current_page = page.copy()  # Make a copy to avoid modifying the original list
            previous_page = pages[idx - 1]
            
            # Split the text of the previous page into sentences
            previous_para_last_sents = re.split(r'(?<=[.?!])\s+', previous_page['page_text'])
            
            # Prepend the last sentence of the previous page to the current page text
            if previous_para_last_sents:  # Check if there are any sentences to prepend
                current_page['page_text'] = previous_para_last_sents[-1] + ' ' + current_page['page_text']
            
            # Append the modified current page to the list
            update_pages.append(current_page)

    return update_pages


def select_pages(pages):
    selected_pages=[]
    pages=pages[2:-2]
    for page in pages:
        if page['token_count']>400:
            selected_pages.append(page)

    return selected_pages


def get_header_n_remove_page_count(elements):
    headers = []
    # Create a new list excluding headers that contain 'page'
    filtered_elements = [dict_ele for dict_ele in elements if not (dict_ele['type'] == 'Header' and 'page' in dict_ele['text'].lower())]

    # Collect headers for counting, now that 'page' headers are excluded
    for dict_ele in filtered_elements:
        if dict_ele['type'] == 'Header':
            headers.append(dict_ele['text'])

    # Use Counter to find the most common header, if headers list is not empty
    if headers:
        counter = Counter(headers)
        header_str, _ = counter.most_common(1)[0]
        print('Title',header_str)
    else:
      header_str=None

    # Replace the most common header text with an empty string in 'Header' elements and change their type
    filtered_elements= [dict_ele for dict_ele in filtered_elements if not (dict_ele['type']=='Header' and dict_ele['text']==header_str)]
    for dict_ele in filtered_elements:
        if dict_ele['type'] == 'Header' and header_str in dict_ele['text']: #to Do handle header  where header_str is not in dict_ele['text']
            dict_ele['text']=dict_ele['text'].replace(header_str,'')  # Consider if you want to keep this step, as it clears the most common header text
            dict_ele['type'] = 'NarrativeText'
        elif dict_ele['type'] == 'Header' and len(dict_ele['text'].split())>5:
          dict_ele['type'] = 'NarrativeText'


    filtered_elements = [ele for ele in filtered_elements if ele['type']!='Header']

    return header_str, filtered_elements




def get_class_default_schema(class_name):
    description=f'doc_chat_class_description_{class_name}'
    default_properties =[{
        'dataType': ['text'],
        'indexFilterable': True,
        'indexSearchable': True,
        'moduleConfig': {'text2vec-openai': {'skip': True,'vectorizePropertyName': False}},
        'name': 'page_number',
        'tokenization': 'word'},
        
        {'dataType': ['text'],
            'indexFilterable': True,
            'indexSearchable': True,
            'moduleConfig': {'text2vec-openai': {'skip': False,'vectorizePropertyName': True}},
            'name': 'content',
            'tokenization': 'word'},
        ]

    class_obj = {
        "class": class_name,
        "description": description,
        "properties": default_properties,
        "moduleConfig": {
            "text2vec-openai": {
                "baseURL": os.getenv("AZURE_OPENAI_ENDPOINT_NP"),
                "deploymentId": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME_NP"),
                "model": "ada",
                "modelVersion": "002",
                "resourceName": 'text-embedding-ada-002',
            }
        },
    }
    return class_obj


def get_failed_objects(results, error_logs, failed_docs):
    if results is None:
        return None
    
    for result in results:
        if result.get('result', {}).get('status') == 'SUCCESS':
            continue

        error_logs.append(result["result"])
        
        content = result.get('properties', {}).pop('content', None)
        meta = result.get('properties', {})
        
        if content is not None:
            doc = Document(page_content=content, metadata=meta)
            failed_docs.append(doc)



def create_and_embed_objects_exp(client, class_name, documents,sleep_seconds):
    # Initialize lists for storing errors and failed document processing results
    failed_docs = []
    error_logs = []

    # Pass failed_docs and error_logs to the callback function
    callback_function = lambda results: get_failed_objects(results, error_logs, failed_docs)

    client.batch.configure(batch_size=50, dynamic=True, creation_time=5, timeout_retries=3, connection_error_retries=3, callback=callback_function)
    with client.batch as batch:
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                weaviate_object = doc.metadata
                weaviate_object["content"] = doc.page_content
            else:
                weaviate_object = doc
                print("Metadata didn't load!")

            try:
                obj_id = batch.add_data_object(data_object=weaviate_object, class_name=class_name)
            except:
                print('Failed to add object')

            
            # Sleep after every 1000 documents, but not at the very first document
            if (i + 1) % 1000 == 0 and i != 0:
                print(f"Processed {i+1} documents, pausing for {sleep_seconds} seconds...")
                sleep(sleep_seconds)
    return failed_docs, error_logs

                
def cal_obj_in_class(class_name,client):
    response = (
    client.query
    .aggregate(class_name)
    .with_meta_count()
    .do()
    ) 
    count=response['data']['Aggregate'][class_name][0]['meta']['count']

    return count

def create_class(class_name, class_schema, client):
    """
    Create a new class based on the provided schema.
    If the class already exists and is not protected, it is recreated.

    Parameters:
    class_name (str): The name of the class to create or recreate.
    class_schema (dict): The schema definition for the class.
    """

    schema = client.schema.get()
    existing_classes =[cls['class'] for cls in schema['classes']]
    # Check if class already exists and is not a protected class
    if class_name in existing_classes and class_name not in PROTECTED_CLASSES and class_name != 'Articles_v2':
        print('Class already exists. Deleting and creating new.')
        client.schema.delete_class(class_name)  # Delete the existing class
        client.schema.create_class(class_schema)  # Create a new class with the provided schema
    else:
        client.schema.create_class(class_schema)  # Create the class as it does not exist
        print(f'New class created with the name: {class_name}')

def read_pdf_pages(file_path):
    file_text=''
    """Reads a PDF file page by page and prints the text content."""

    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            file_text+=text
            #print(f"Page {page_num + 1}:\n{text}\n")
    return file_text

def process_pdf(file_path):
    elements, _ = parse_pdf_w_unstructured(file_path, True)
    doc_title, elements = get_header_n_remove_page_count(elements)
    pages=page_chunking(elements)
    return pages

## Creating classes
def ingestion_documents_df(file_path, client):

    class_name, _ = os.path.splitext(os.path.basename(file_path))
    class_name=class_name.replace(' ','_')
    class_name=class_name.capitalize()

    print('processing file:', file_path)
    pages=process_pdf(file_path)
    chunks=create_lg_docs(pages)

    class_schema=get_class_default_schema(class_name)
    create_class(class_name, class_schema, client)
    missed_docs,error_logs=create_and_embed_objects_exp(client, class_name, chunks,sleep_seconds=20)
    """if len(error_logs):
        missed_docs,error_logs=create_and_embed_objects_exp(client, class_name, missed_docs,sleep_seconds=20)
    assert len(error_logs)==0
    assert len(pages)==cal_obj_in_class(class_name, client)"""

    return pages, class_name


def hybrid_search(client, class_name, question, top_k):
    question = re.sub(r"[^a-zA-Z0-9\s]", "", question)
    response = (
    client.query
    .get(class_name, ["content", "page_number"])  #Feilds to return in response
    .with_hybrid(
        query=question,
        alpha=0.5,
        properties=["content"], # Restrict the Keyword search to these property only
        fusion_type=HybridFusion.RELATIVE_SCORE 
    ).with_additional(['score'])
    .with_limit(top_k)#.with_autocut(1)
    .do())

    retrived_docs = [result['content'] for result in response['data']['Get'][class_name]]
    page_numbers = [result['page_number'] for result in response['data']['Get'][class_name]]

    docs=[]
    for idx in range(0,len(retrived_docs)):
        page = Document(page_content=retrived_docs[idx],
        metadata = {'page_number':page_numbers[idx]})
        docs.append(page)

    return docs 

def format_docs(docs):
        # Assuming you have XML tags named <document> and </document>
    xml_tag_start = "<document>"
    xml_tag_end = "</document>"
    xml_doc_id_start = "<doc_page_number>"
    xml_doc_id_end = "</doc_page_number>"
    top_k_content_with_tags = [f"{xml_doc_id_start}{doc.metadata['page_number']}{xml_doc_id_end}{xml_tag_start}{doc.page_content}{xml_tag_end}" for idx, doc in enumerate(docs)]
    content='\n\n'.join(top_k_content_with_tags)
    return content