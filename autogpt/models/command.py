from typing import Any, Callable, Dict, Optional

from langchain.tools import BaseTool

from autogpt.config import Config
from autogpt.logs import logger


class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        signature (str): The signature of the function that the command executes. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., Any],
        signature: Dict[str, Dict[str, Any]],
        enabled: bool | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.signature = signature
        self.enabled = enabled
        self.disabled_reason = disabled_reason

    def __call__(self, *args, **kwargs) -> Any:
        if hasattr(kwargs, "config") and callable(self.enabled):
            self.enabled = self.enabled(kwargs["config"])
        if not self.enabled:
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}, args: {self.signature}"

    @classmethod
    def generate_from_langchain_tool(
        cls, tool: BaseTool, arg_converter: Optional[Callable] = None
    ) -> "Command":
        # Change `title` to `name` in the arg definitions
        command_args = {}
        for name, arg in tool.args.items():
            command_args[name] = {"name": arg.pop("title"), **arg}

        def wrapper(*args, **kwargs):
            # a Tool's run function doesn't take an agent as an arg, so just remove that
            agent = kwargs.pop("agent")

            # Allow the command to do whatever arg conversion it needs
            if arg_converter:
                tool_input = arg_converter(kwargs, agent)
            else:
                tool_input = kwargs

            logger.debug(f"Running LangChain tool {tool.name} with arguments {kwargs}")

            return tool.run(tool_input=tool_input)

        command = cls(
            name=tool.name,
            description=tool.description,
            method=wrapper,
            signature=command_args,
        )

        # Avoid circular import
        from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER

        # Set attributes on the command so that our import module scanner will recognize it
        setattr(command, AUTO_GPT_COMMAND_IDENTIFIER, True)
        setattr(command, "command", command)

        return command
