from langchain.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain.prompts import PromptTemplate

def answer_gen_runnable(llm):
    # - End your response with final answer in the form <ANSWER>: $"A succint answer to the question based on the context, in the persona of representative assistant from morningstar help." and <Supporting_doc_src_id>: Mention onkly the doc ids that supports the answer.

    # single_content=flatten_list[idx].page_content
    system_gen_temp_1 = """### You are helpful AI assistant you are equipped to answer questions by referencing to provided documents and given guidelines."""

    human_temp_2 = """Here are the inputs, answer based on context: \nQuestion: {question}\nContext(by MorningStar financial service): {context}\n\n
    ### Important: Once provide the answer also mention the source page number of the doc that supports the answer.
    ### Output format: <Answer>:"write your answer here"</Answer>\n\n <supporting_doc_page_number>: "mention doc page_number that supports the answer" </supporting_doc_page_number>"""

    human_temp_1 = """Here are the inputs, answer based on context: \nQuestion: {question}\nContext(by MorningStar financial service): {context}\n\n
    ### Here is guidelines to pay attention to:
    - First, understand the question and thoroughly analyze the given context, then provide step-by-step reasoning for answering the question based on the information from the context.    
    - While describing the reasoning, if requires you can copy past facts/supporting sentences from the context on which the resoning is based and include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
    - End your response with final answer in the form <ANSWER>: $"Enter a well formed and compreshensive natural language answer to the question" </ANSWER> , <supporting_doc_page_number>: "mention doc page_number that supports the answer" </supporting_doc_page_number>
    - Simply respond with "SORRY DONT HAVE RELEVANT INFORMATION.", incase the refernceing docs lacks sufficent info to answer."""

    prompt = """Here are the inputs, answer based on context: \nQuestion: {question}\nContext(by MorningStar financial service): {context}\n\n
       ### Important: Once provide the answer also mention the source page number of the doc that supports the answer.
       ### Output format: <Answer>:"write your answer here"</Answer>\n\n <supporting_doc_page_number>: "mention doc page_number that supports the answer" </supporting_doc_page_number>"""

    # sys_temp = SystemMessagePromptTemplate.from_template([system_gen_temp_1])
    # huma_temp= HumanMessagePromptTemplate.from_template([human_temp_2])
    # answer_gen_prompt = ChatPromptTemplate.from_messages([sys_temp, huma_temp])
    output_parser = StrOutputParser()
    prompt = """Question: {question}\nContext(by MorningStar financial service): {context}\n\nPlease answer above questions based only on the given context. Focus on providing an overview of main points, highlighting key trends, performances, and notable conclusions. Include specific sections and sub-sections with references to pages used. Ensure the answer is concise and captures the most important information"""
    prompt = PromptTemplate(input_variables=['question', 'context'], template=prompt)
    runnable = prompt | llm | output_parser

    return runnable


def conversation_runnable(llm):
    sys_temp="Based on the provided conversation history between a user and an AI assistant, and a follow-up question, your task is to formulate a standalone question. A standalone question is a clear and self-contained rephrasing of the original query, enhanced with contextual insights derived from the chat history."""
    
    human_temp="""Given the conversation history below and a follow-up input, rephrase the question to create a standalone version that clearly conveys what the user is asking, incorporating relevant context from the chat history if necessary. 

    Chat History:
    {chat_history}

    Follow Up Input:
    {question}

    Standalone Question:"""

    sys_temp=SystemMessagePromptTemplate.from_template(sys_temp)
    prompts_list=[sys_temp]
    if human_temp:
        human_temp=SystemMessagePromptTemplate.from_template(human_temp)
        prompts_list.append(human_temp)
    
    prompt=ChatPromptTemplate.from_messages(prompts_list)
    #request_size = request_token_sizes[summary_level]
    output_parser = StrOutputParser()

    run = prompt | llm  | output_parser
    return run


def summary_runnable(llm):
    system_gen_temp_1 = """You are an expert financial researcher.Your task is to read through the entire document and provide a concise yet comprehensive summary \
    so that it would be helpful to your fellow researchers and investors.
    The goal is to capture the essence of the document, ensuring that no significant information is overlooked.
    """

    human_temp_2= "Here is the input document, generate the summary according to give instructions: {document}"


    sys_temp = SystemMessagePromptTemplate.from_template([system_gen_temp_1])
    huma_temp= HumanMessagePromptTemplate.from_template([human_temp_2])
    summary_prompt = ChatPromptTemplate.from_messages([sys_temp,huma_temp])

    output_parser = StrOutputParser()

    runnable=summary_prompt | llm | output_parser

    return runnable

