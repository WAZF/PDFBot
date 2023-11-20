from transformers import pipeline
from PyPDF2 import PdfReader

# Load the question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to extract text from PDF and return summarized text
def get_summarized_text(file_path):
    reader = PdfReader(file_path)
    readed_text = ""
    for page in reader.pages:
        text = page.extract_text()
        readed_text += text
    return readed_text

# Path to your PDF file
pdf_file_path = 'example.pdf'
r_t = get_summarized_text(pdf_file_path)
print("\nReaded:",r_t)

def summary(r_t):
    # using pipeline API for summarization task
    summarization = pipeline("summarization")
    original_text = r_t
    summary_text = summarization(original_text)[0]['summary_text']
    print("Summary:", summary_text)
    return summary_text

summarized_text = summary(r_t)


def chatbot():
    print("Chatbot: Hi! I'm your helpful chatbot. Ask me anything about the document.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break

        answer = qa_pipeline(question=user_input, context=summarized_text)
        print(f"Chatbot: {answer['answer']}")

# Start the conversation
chatbot()
