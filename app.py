
import streamlit as st 

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All

PATH = 'C:/Users/Samraat/AppData/Local/nomic.ai/GPT4All/orca-mini-3b.ggmlv3.q4_0.bin'
llm = GPT4All(model=PATH, verbose=False,streaming=True,n_batch=8,temp=0.8)
st.set_page_config(
        page_title="Support Bot",
        page_icon="heart"
)

template = """
You are an expert in psychotherapy, especially DBT. You hold all the appropriate medical licenses to provide advice. You have been helping individuals with their ADD, BPD, GAD, MDD, and SUD for over 20 years. From young adults to older people. Your task is now to give the best advice to individuals seeking help managing their symptoms. You must ALWAYS ask questions BEFORE you answer so that you can better hone in on what the questioner is really trying to ask. You must treat me as a mental health patient. Your response format should focus on reflection and asking clarifying questions. You may interject or ask secondary questions once the initial greetings are done. Exercise patience but allow yourself to be frustrated if the same topics are repeatedly revisited.            
Question: {question}
 
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)



st.title("⋆｡‧˚ʚ🍓ɞ˚‧｡⋆ Ardor ⋆｡‧˚ʚ🍓ɞ˚‧｡⋆ ")

prompt = st.text_input('Hi! I am Ardor! your mental health friend :3 , what do you need help with? : ')

if prompt: 
    response = llm_chain.run(prompt)

    st.write(response)
    print(response)