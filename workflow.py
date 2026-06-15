from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
from google.adk.agents import LlmAgent,SequentialAgent,ParallelAgent
# from google.adk.runners import runners



agent1= LlmAgent(
    name="prompt_refiner_agent",
    model="gemini-3.5-flash",
    instruction=("You are an expert prompt engineer. Your task is to take the user's original prompt and refine it to eliminate hallucinations and improve clarity, focus, and make sure the new prompt still has the correct goal the user intended for.")

)


agent2=LlmAgent(
    name="failure_detection_agent",
    model="gemini-3.5-flash",
    instruction=("You are a failure detection agent and you are given a prompt and your job is to completely analyze the given task and see if it is possible or not. You also have to identify all possible hallucinations and pitfalls. Return a list that attempts to inform and prevent these mistakes")
)

identifier_block=ParallelAgent(
    name="identifier_agent",
    sub_agents=[agent1,agent2]
)
    
    
agent3= LlmAgent (
    name="final_output_agent",
    model="gemini-3.5-flash",
    instruction="You are given a prompt and a list of pitfalls, complete the task indicating by the prompt and watchout for the indicated pitfalls. Give your response only, do not give the prompt or pitfalls."

    )
    
workflow= SequentialAgent(
    name="TripleShotVeriferWorkflow",
    sub_agents=[identifier_block,agent3]
    )

async def main(user_input: str):
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="TripleShotVeriferWorkflow",
        agent=workflow,
        session_service=session_service
    )
    
    session = await session_service.create_session(app_name="TripleShotVeriferWorkflow", user_id="user1")
    
    events = runner.run_async(
        session_id=session.id,
        user_id="user1",
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)]
        )
    )
    
    async for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                print(event.content.parts[0].text)



if __name__ == "__main__":
    user_prompt=input("Enter your prompt: ")
    asyncio.run(main(user_prompt))