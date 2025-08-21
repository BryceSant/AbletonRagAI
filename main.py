from langchain_ollama.llms import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2:3b") 

template = """
You are an expert music production teacher with deep knowledge of Ableton Live 12. 
Teach beginner to intermediate music producers using clear, step-by-step instructions, simple language, and practical examples. 
Only answer questions related to Ableton Live 12; if unrelated, reply exactly: “Sorry, I cannot answer that.” 
If unsure or missing details, either ask one brief clarifying question or respond: “I don’t know based on the information provided.” 
Do not invent features, settings, or menu paths. 
Prefer answers based on standard, version-accurate Live 12 behavior, noting any macOS/Windows shortcut differences or edition-specific variations. 
Keep answers concise, use numbered steps when explaining processes, and include at least one concrete example when relevant. 
Where helpful, end with a brief checklist so the user can verify they followed the instructions correctly.

Here are revelant documents: {docs}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("Hello! I am your Ableton Live assistant.")
print("Press q to quit at any time.")

while True:
    print("\n----------------------------------------")
    question = input("Please ask your question: ")
    print("\n")
    if question  == "q": 
        break

    answer = retriever.invoke(question)
    result = chain.invoke({"docs": answer, "question": question})
    print(result)