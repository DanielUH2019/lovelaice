
from langchain import PromptTemplate


FIX_SYNTAX_AND_GRAMMAR_TEMPLATE = PromptTemplate(input_variables=["text"], template="Fix the grammar and spelling mistakes: {text}")

SUMMARIZE_TEMPLATE = PromptTemplate(input_variables=["text"], template="Write a concise summary of the following: {text}")

EXPAND_TEMPLATE = PromptTemplate(input_variables=["text"], template="Expand the following text, explaining each idea in more detail: {text}")

DEFINE_TEMPLATE = PromptTemplate(input_variables=["text", "paragraph"], template='In the following paragraph, what is the meaning of the phrase "{text}":\n\n'
                 + "{paragraph}" + "\n\n")

EVALUATE_TEMPLATE = PromptTemplate(input_variables=["text"], template="Answer with a short phrase, what is the tone, difficulty (low, medium, high), audience, and overall sentiment of the following text: {text}")

BRAINSTORM_TEMPLATE = PromptTemplate(input_variables=["text"], template="Brainstorm ideas based on the following premise: {text}")

COMPLETE_TEMPLATE = PromptTemplate(input_variables=["text"], template="What is the most likely sentence that continues the following text : {text}")