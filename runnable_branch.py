from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text in less than 100 words \n {text}',
    input_variables=['text']
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

summary_chain = RunnableSequence(
    RunnableLambda(lambda x: {"text": x}),
    prompt2,
    model,
    parser
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, summary_chain),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))