import os

from autogpt.core.agent import SampleAgent
from autogpt.core.messaging.queue_channel import QueueChannel


def build_agent(name: str, channel: QueueChannel):
    """Build the agent."""
    print("Building agent...")
    # get api key from env
    api_key = os.getenv("OPENAI_API_KEY")
    agent_str = f"""
        {{
            "uid": "test_agent",
            "name": "test_agent",
            "llm_provider": {{
                "type": "OpenAIProvider",
                "api_key": "ccc",
                "chat_completion_model": "gpt-3.5-turbo"
            }},
            "message_broker": {{
                "type": "MessageBroker",
                "channels": []
            }}
        }}
    """

    agent = SampleAgent.parse_raw(agent_str)
    agent.message_broker.add_channel(channel)
    return agent
