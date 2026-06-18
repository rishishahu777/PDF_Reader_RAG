from groq import Groq

client = Groq(
    api_key="YOUR_GROQ_API_KEY"
)


def generate_answer(
    context,
    question
):

    prompt = f"""
You are a PDF Question Answering Assistant.

Answer ONLY from the context.

If the answer is not found in the context,
reply exactly:

I cannot find this information in the document.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
