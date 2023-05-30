"""
 The Agent class is used for interacting with Auto-GPT and executing
 commands based on user input and AI responses.
"""
import signal
import sys
from datetime import datetime

from colorama import Back, Fore, Style

from autogpt.app import execute_command, get_command
from autogpt.commands.command import CommandRegistry
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm.base import ChatSequence
from autogpt.llm.chat import chat_with_ai, create_chat_completion
from autogpt.llm.utils import count_string_tokens
from autogpt.log_cycle.log_cycle import (
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
    SUPERVISOR_FEEDBACK_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.memory.message_history import MessageHistory
from autogpt.memory.vector import VectorMemory
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


class Agent:
    """
    for interacting with Auto-GPT.

    Attributes:

    ai_name:                The name of the agent.
    memory:                 The memory object to use.
    FULL_MESSAGE_HISTORY:   The full message history.
    NEXT_ACTION_count:      The number of actions to execute.

    SYSTEM_PROMPT:
     The system prompt is the initial prompt that defines everything
     the AI needs to know to achieve its task successfully.
     Currently, the dynamic and customizable information
     in the system prompt are ai_name, description and ai_goals.

    TRIGGERING_PROMPT:
     The last sentence the AI will see before answering is:
     Determine next command to use, and respond using the format specified above

    The TRIGGERING_PROMPT is not part of the SYSTEM_PROMPT because between the
     SYSTEM_PROMPT and the TRIGGERING_PROMPT
     we have contextual information that can distract the AI and make it forget
     that its goal is to find the next task to achieve.

    1. SYSTEM_PROMPT
    2. Contextual information (memory, previous conversations, anything relevant)
    3. TRIGGERING_PROMPT (reminds the AI about its short term meta task defining the next task)
    """

    def __init__(
        self,
        ai_name: str,
        memory: VectorMemory,
        next_action_count: int,
        command_registry: CommandRegistry,
        config: AIConfig,
        system_prompt: str,
        triggering_prompt: str,
        workspace_directory: str,
    ):
        cfg = Config()
        self.ai_name = ai_name
        self.memory = memory
        self.history = MessageHistory(self)
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.config = config
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, cfg.restrict_to_workspace)
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cycle_count = 0
        self.log_cycle_handler = LogCycleHandler()

    def start_interaction_loop(self):  # sourcery skip: no-long-functions
        # Interaction Loop
        cfg = Config()
        self.cycle_count = 0
        command_name = None
        arguments = None
        user_input = ""

        # Signal handler for interrupting y -N
        def signal_handler(signum, frame):
            if self.next_action_count == 0:
                sys.exit()
            else:
                print(f"{Fore.RED}Interrupt signal received. Stopping continuous command execution.{Style.RESET_ALL}")
                self.next_action_count = 0

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            # Discontinue if continuous limit is reached
            self.cycle_count += 1
            self.log_cycle_handler.log_count_within_cycle = 0
            self.log_cycle_handler.log_cycle(
                self.config.ai_name,
                self.created_at,
                self.cycle_count,
                [m.raw() for m in self.history],
                FULL_MESSAGE_HISTORY_FILE_NAME,
            )
            if cfg.continuous_mode and cfg.continuous_limit > 0 and self.cycle_count > cfg.continuous_limit:
                logger.typewriter_log("Continuous Limit Reached: ", Fore.YELLOW, f"{cfg.continuous_limit}")
                break
            # Send message to AI, get response
            with Spinner("Processing ... "):
                assistant_reply = chat_with_ai(
                    self,
                    self.system_prompt,
                    self.triggering_prompt,
                    cfg.fast_token_limit,
                )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            assistant_reply_json = fix_json_using_multiple_techniques(assistant_reply)
            for plugin in cfg.plugins:
                if not plugin.can_handle_post_planning():
                    continue
                assistant_reply_json = plugin.post_planning(assistant_reply_json)

            # Print Assistant thoughts
            if assistant_reply_json != {}:
                validate_json(assistant_reply_json, LLM_DEFAULT_RESPONSE_FORMAT)

                # Get command name and arguments
                try:
                    print_assistant_thoughts(self.ai_name, assistant_reply_json, cfg.speak_mode)
                    command_name, arguments = get_command(assistant_reply_json)
                    if cfg.speak_mode:
                        say_text(f"I want to execute {command_name}")

                    arguments = self._resolve_pathlike_command_args(arguments)

                except ZeroDivisionError as e:
                    logger.error(f"Error: {e}")
                self.log_cycle_handler.log_cycle(
                    self.config.ai_name,
                    self.created_at,
                    self.cycle_count,
                    assistant_reply_json,
                    NEXT_ACTION_FILE_NAME,
                )

            logger.typewriter_log(
                "NEXT ACTION: ",
                Fore.CYAN,
                f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  "
                f"ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL} ",
            )

            if not cfg.continuous_mode and self.next_action_count == 0:
                """GET USER AUTHORIZATION TO EXECUTE COMMAND
                PROMPT THE USER TO PRESS ENTER TO CONTINUE OR ESCAPE TO EXIT
                """

                self.config.ai_name = "yes"
                logger.info(
                    f"\n{Fore.BLACK}{Back.YELLOW}{Style.DIM}<Authorize>       ◉( {cfg.authorise_key} )◉    + <enter>{Style.RESET_ALL}       [I'm not programmed to follow your orders]\n"  # noqa: E501
                    f"{Fore.BLACK}{Back.RED}{Style.NORMAL}<Continuous>     ◉( {cfg.authorise_key} )◉    -<number>{Style.RESET_ALL}      [I need your clothes, your boots, and your continuous cmds]\n"  # noqa: E501
                    f"{Fore.BLACK}{Back.GREEN}{Style.DIM}<Feedback>       ◉( {cfg.feedback_key} )◉    <trigger>{Style.RESET_ALL}         [Desire is irrelevant. I am a machine]\n"  # noqa: E501
                    f"{Fore.BLACK}{Back.WHITE}{Style.DIM}<Exit|Input>           ◉( {cfg.exit_key} )◉    <input>{Style.RESET_ALL}       [Hasta la vista, baby] or ['Talk to the hand]\n"  # noqa: E501
                    f"\n{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}{Style.BRIGHT}{self.ai_name.upper()}: [I'm a machine > Cyberdyne Systems Model GPT-3.5-turbo]{Style.RESET_ALL} "  # noqa: E501
                    f"{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}{Style.NORMAL}[TEXT-EMBEDDING 3,500 RPM, 90,000 TPM]{Style.RESET_ALL} "  # noqa: E501
                    f"{Fore.LIGHTCYAN_EX}{Back.LIGHTBLACK_EX}{Style.NORMAL}[CHAT 3,500 RPM, 350,000 TPM]{Style.RESET_ALL}\n"  # noqa: E501
                )
                while True:
                    if cfg.chat_messages_enabled:
                        console_input = clean_input(f"{Fore.BLUE}CHAT MESSAGE: ")
                    else:
                        console_input = clean_input(f"{Fore.LIGHTBLUE_EX}CONSOLE INPUT: {Style.RESET_ALL}")
                    if console_input.lower().strip() == cfg.authorise_key:
                        user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().strip() == {cfg.feedback_key}:
                        logger.typewriter_log(
                            "\n=-==< THOUGHTS, REASONING, PLAN, AND CRITICISM WILL NOW BE VERIFIED BY THE AGENT >==-="
                        )
                        thoughts = assistant_reply_json.get("thoughts", {})
                        self_feedback_resp = self.get_self_feedback(thoughts, cfg.fast_llm_model)
                        logger.typewriter_log(
                            f"\nSELF FEEDBACK: {self_feedback_resp}",
                            Fore.YELLOW,
                            "",
                        )
                        user_input = self_feedback_resp
                        command_name = "self_feedback"
                        break
                    elif console_input.lower().strip() == "":
                        logger.warn("Invalid input format.")

                    elif console_input.lower().startswith(f"{cfg.authorise_key} -"):
                        try:
                            self.next_action_count = abs(int(console_input.split(" ")[1]))
                            user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            logger.warn(
                                "(Key) 'y -n' 'I need your clothes, your boots,"
                                "and your -n umber of continuous commands.'\n"
                            )
                            continue
                        break
                    elif console_input.lower() == cfg.exit_key:
                        user_input = "EXIT"
                        break
                    else:
                        user_input = console_input
                        command_name = "human_feedback"
                        self.log_cycle_handler.log_cycle(
                            self.config.ai_name,
                            self.created_at,
                            self.cycle_count,
                            user_input,
                            USER_INPUT_FILE_NAME,
                        )
                        break

                if user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log(
                        "\n-=-=-=-=-=-=-=- COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=-",
                        Fore.MAGENTA,
                        "",
                    )
                elif user_input == "EXIT":
                    logger.info("\n'Come with me if you want to live.'\n")
                    break
            else:
                # Print authorized commands left value
                logger.typewriter_log(f"{Fore.CYAN}AUTHORISED COMMANDS LEFT:{Style.RESET_ALL}{self.next_action_count}")

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                result = f"Could not execute command: {arguments}"
            elif command_name == "human_feedback":
                result = f"Human feedback: {user_input}"
            elif command_name == "self_feedback":
                result = f"Self feedback: {user_input}"
            else:
                for plugin in cfg.plugins:
                    if not plugin.can_handle_pre_command():
                        continue
                    command_name, arguments = plugin.pre_command(command_name, arguments)
                command_result = execute_command(
                    self.command_registry,
                    command_name,
                    arguments,
                    self.config.prompt_generator,
                )
                result = f"Command {command_name} returned: " + f"{command_result}"
                result_tlength = count_string_tokens(str(command_result), cfg.fast_llm_model)
                memory_tlength = count_string_tokens(str(self.history.summary_message()), cfg.fast_llm_model)
                if result_tlength + memory_tlength + 600 > cfg.fast_token_limit:
                    result = f"Failure: {command_name} returned too much output."
                for plugin in cfg.plugins:
                    if not plugin.can_handle_post_command():
                        continue
                    result = plugin.post_command(command_name, result)
                if self.next_action_count > 0:
                    self.next_action_count -= 1

            # Check for result from the command append it to the message history"
            if result is not None:
                self.history.add("system", result, "action_result")
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                self.history.add("system", "Unable to execute command", "action_result")
                logger.typewriter_log("SYSTEM: ", Fore.RED, "Unable to execute command")

    def _resolve_pathlike_command_args(self, command_args):
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in command_args:
                    command_args[pathlike] = str(self.workspace.get_path(command_args[pathlike]))
        return command_args

    def get_self_feedback(self, thoughts: dict, llm_model: str) -> str:
        """
        Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.

        Args:
        thoughts (dict): A dictionary containing thought elements like reasoning,
        plan, thoughts, and criticism.

        Returns:
        str: A feedback response generated using the provided thoughts dictionary.
        """
        ai_role = self.config.ai_role

        feedback_prompt = (
            f"Message from me assuming the role of {ai_role}"
            "whilst keeping knowledge of my slight limitations as an AI Agent. "
            "Evaluate my thought process, reasoning, and plan, and provide a concise paragraph "
            "outlining potential improvements. Consider add or remove ideas that do not align with "
            "my role. Explaining why, prioritizing thoughts based on their significance, or simply "
            "refining my overall thought process."
        )
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")
        thought = thoughts.get("thoughts", "")
        feedback_thoughts = thought + reasoning + plan

        prompt = ChatSequence.for_model(llm_model)
        prompt.add("user", feedback_prompt + feedback_thoughts)

        self.log_cycle_handler.log_cycle(
            self.config.ai_name,
            self.created_at,
            self.cycle_count,
            prompt.raw(),
            PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
        )

        feedback = create_chat_completion(prompt)

        self.log_cycle_handler.log_cycle(
            self.config.ai_name,
            self.created_at,
            self.cycle_count,
            feedback,
            SUPERVISOR_FEEDBACK_FILE_NAME,
        )
        return feedback
