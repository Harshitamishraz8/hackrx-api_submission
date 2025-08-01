import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")

client = Groq(api_key=groq_api_key)

def query_llm(questions: list, contexts: list) -> list:
    """Generate answers using Groq LLM"""
    if not contexts:
        return ["No relevant information found in the document."] * len(questions)
    
    # Combine contexts (limit to top 5 to stay within token limits)
    combined_context = "\n\n".join(contexts[:5])
    
    answers = []
    
    for question in questions:
        try:
            prompt = f"""You are a helpful assistant that answers questions based on the provided document context. 
Answer the question using only the information from the context below. If the information is not available in the context, say "Information not available in the provided document."

Context:
{combined_context}

Question: {question}

Answer:"""

            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=500,
                top_p=1,
                stop=None
            )
            
            answer = response.choices[0].message.content.strip()
            answers.append(answer)
            print(f"Generated answer for '{question}': {answer[:100]}...")
            
        except Exception as e:
            print(f"Error generating answer for question '{question}': {e}")
            answers.append("Error generating answer. Please try again.")
    
    return answers