import asyncio

from autogpt.core.messaging.messages import ShutdownMessage
from autogpt.core.messaging.queue_channel import QueueChannel
from autogpt.core.schema import BaseAgent
from autogpt.core.server.models.agent import *


async def create_agent(agent: BaseAgent, queue_channel: QueueChannel) -> asyncio.Task:
    if len(agent.message_broker.channels) == 0:
        agent.message_broker.add_channel(queue_channel)

    return asyncio.create_task(agent.run())


async def stop_agent(
    stop_request: StopAgentReq, message_queue: QueueChannel, task: asyncio.Task
) -> StopAgentResponse:
    """Sends a stop request to the agent"""
    if stop_request.immediately:
        task.cancel()
    else:
        stop = ShutdownMessage(
            from_uid="user",
            to_uid=stop_request.agent_id,
            timestamp=0,
            immediately=stop_request.immediately,
        )
        await message_queue.send(stop)
    return StopAgentResponse(agent_id=stop_request.agent_id, status="Success")
