import logging
import time
from typing import Optional
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_llm(timeout: int = 120) -> Optional[Ollama]:
    """Initialize the Ollama LLM with error handling"""
    try:
        logger.info("Initializing Ollama with llama2 model...")
        llm = Ollama(model="llama2", timeout=timeout)

        llm.invoke("test")
        logger.info("Ollama connection successful")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {str(e)}")
        print("\nError: Could not connect to Ollama. Please make sure:")
        print("1. Ollama is installed and running")
        print("2. The llama2 model is pulled (run: ollama pull llama2)")
        return None

def validate_input(topic: str) -> bool:
    """Validate the input topic"""
    if not topic or len(topic.strip()) < 2:
        print("\nError: Please enter a valid topic (at least 2 characters)")
        return False
    return True

def create_study_guide(topic: str, llm: Ollama) -> Optional[str]:
    """Create a study guide with error handling"""
    try:
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""Create a concise study guide about {topic}. Include:
            1. Key concepts
            2. Important points
            3. Simple examples
            
            Keep it brief and clear."""
        )
        
        chain = prompt | llm
        logger.info(f"Generating study guide for topic: {topic}")
        start_time = time.time()
        
        study_guide = chain.invoke({"topic": topic})
        
        end_time = time.time()
        logger.info(f"Study guide generated in {end_time - start_time:.2f} seconds")
        return study_guide
        
    except Exception as e:
        logger.error(f"Error generating study guide: {str(e)}")
        print("\nError: Failed to generate study guide. Please try again.")
        return None

def main():
    """Main function to run the study assistant"""
    llm = initialize_llm()
    if not llm:
        return
    
    while True:
        topic = input("\nEnter the topic you want to study (or 'quit' to exit): ").strip()
        
        if topic.lower() == 'quit':
            print("\nGoodbye!")
            break
            
        if not validate_input(topic):
            continue
            
        study_guide = create_study_guide(topic, llm)
        if study_guide:
            print("\nStudy Guide:")
            print(study_guide)

if __name__ == "__main__":
    main() 