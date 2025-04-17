from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

# Initialize Ollama with llama2 model
llm = Ollama(model="llama2")

# Create a simpler prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Create a concise study guide about {topic}. Include:
    1. Key concepts
    2. Important points
    3. Simple examples
    
    Keep it brief and clear."""
)

# Create chain using RunnableSequence
chain = prompt | llm

def study_assistant(topic):
    """Run the study assistant with the given topic"""
    print("\nCreating study guide... This should take less than a minute...")
    # Get the study guide directly
    study_guide = chain.invoke({"topic": topic})
    return study_guide

if __name__ == "__main__":
    topic = input("Enter the topic you want to study: ")
    result = study_assistant(topic)
    print("\nStudy Guide:")
    print(result) 