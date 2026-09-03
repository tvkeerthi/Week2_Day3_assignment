# main.py
#
# Sends a prompt to an OpenAI model via LangChain, prints the response,
# and traces the run in LangSmith (tracing is enabled purely via
# environment variables below — no extra code needed for that part).

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load API keys from the .env file into environment variables
load_dotenv()

def main():
    # LangSmith tracing reads these env vars automatically once they're set
    # (see .env.example below) — nothing else needed to enable tracing.

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )

    prompt = "Give me one interesting fact about space in a single sentence."

    print(f"Prompt: {prompt}\n")

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()