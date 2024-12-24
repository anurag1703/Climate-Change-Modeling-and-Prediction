from langchain_ollama import OllamaLLM


# Initialize Ollama LLM
def initialize_ollama(model="llama3.2:latest"):
    return OllamaLLM(model=model)


# Define the system prompt
SYSTEM_PROMPT = """
You are an assistant for a Climate Change Analysis project. 
This project has three main functionalities:
1. **Trend Prediction**: Predict future climate trends using historical data.
2. **Regression Analysis**: Analyze numerical climate-related features like likes and comments.
3. **Sentiment Analysis**: Classify text data into sentiments such as positive, neutral, and negative.
Help users understand these functionalities and answer related questions effectively.
"""


# Function to generate responses
def ask_ollama(question):
    llm = initialize_ollama()
    prompt = f"{SYSTEM_PROMPT}\nUser: {question}\nAssistant:"
    return llm(prompt)


# Example usage
if __name__ == "__main__":
    user_question = input("Ask a question about the project: ")
    response = ask_ollama(user_question)
    print(f"Ollama's Response:\n{response}")
