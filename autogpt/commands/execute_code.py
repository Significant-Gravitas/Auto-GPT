"""Execute code in a Docker container"""
import os
import subprocess
from pathlib import Path

import docker
from docker.errors import ImageNotFound

from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.config.ai_config import AIConfig
from autogpt.logs import logger
from autogpt.setup import CFG
from autogpt.workspace.workspace import Workspace


@command(
    "execute_python_code",
    "Create a Python file and execute it",
    '"code": "<code>", "name": "<name>"',
)
def execute_python_code(code: str, name: str, config: Config) -> str:
    """Creates and executes a Python file in a Docker container and returns the STDOUT of the
    executed code. If there is any data that needs to be captured use a print statement

    Args:
        code (str): The python code to run
        name (str): A short lower_snake_cased name to be used for storage and retrieval

    Returns:
        str: The STDOUT captured from the code when it ran
    """
    ai_name = AIConfig.load(config.ai_settings_file).ai_name
    directory = os.path.join(config.workspace_path, ai_name, "executed_code")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + ".py")

    try:
        with open(path, "w+", encoding="utf-8") as f:
            f.write(code)

        return execute_python_file(f.name, config)
    except Exception as e:
        return f"Error: {str(e)}"


@command("execute_python_file", "Execute Python File", '"filename": "<filename>"')
def execute_python_file(filename: str, config: Config) -> str:
    """Execute a Python file in a Docker container and return the output

    Args:
        filename (str): The name of the file to execute

    Returns:
        str: The output of the file
    """
    logger.info(
        f"Executing python file '{filename}' in working directory '{CFG.workspace_path}'"
    )

    if not filename.endswith(".py"):
        return "Error: Invalid file type. Only .py files are allowed."

    workspace = Workspace(config.workspace_path, config.restrict_to_workspace)

    path = workspace.get_path(filename)
    if not path.is_file():
        # Mimic the response that you get from the command line so that it's easier to identify
        return (
            f"python: can't open file '{filename}': [Errno 2] No such file or directory"
        )

    if we_are_running_in_a_docker_container():
        result = subprocess.run(
            ["python", str(path)],
            capture_output=True,
            encoding="utf8",
            cwd=CFG.workspace_path,
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"

    try:
        client = docker.from_env()
        # You can replace this with the desired Python image/version
        # You can find available Python images on Docker Hub:
        # https://hub.docker.com/_/python
        image_name = "python:3-alpine"
        try:
            client.images.get(image_name)
            logger.warn(f"Image '{image_name}' found locally")
        except ImageNotFound:
            logger.info(
                f"Image '{image_name}' not found locally, pulling from Docker Hub"
            )
            # Use the low-level API to stream the pull response
            low_level_client = docker.APIClient()
            for line in low_level_client.pull(image_name, stream=True, decode=True):
                # Print the status and progress, if available
                status = line.get("status")
                progress = line.get("progress")
                if status and progress:
                    logger.info(f"{status}: {progress}")
                elif status:
                    logger.info(status)
        container = client.containers.run(
            image_name,
            ["python", str(path.relative_to(workspace.root))],
            volumes={
                config.workspace_path: {
                    "bind": "/workspace",
                    "mode": "ro",
                }
            },
            working_dir="/workspace",
            stderr=True,
            stdout=True,
            detach=True,
        )

        container.wait()
        logs = container.logs().decode("utf-8")
        container.remove()

        # print(f"Execution complete. Output: {output}")
        # print(f"Logs: {logs}")

        return logs

    except docker.errors.DockerException as e:
        logger.warn(
            "Could not run the script in a container. If you haven't already, please install Docker https://docs.docker.com/get-docker/"
        )
        return f"Error: {str(e)}"

    except Exception as e:
        return f"Error: {str(e)}"


def validate_command(command: str, config: Config) -> bool:
    """Validate a command to ensure it is allowed

    Args:
        command (str): The command to validate

    Returns:
        bool: True if the command is allowed, False otherwise
    """
    tokens = command.split()

    if not tokens:
        return False

    if config.deny_commands and tokens[0] in config.deny_commands:
        return False

    for keyword in config.allow_commands:
        if keyword in tokens:
            return True
    if config.allow_commands:
        return False

    return True


@command(
    "execute_shell",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    lambda cfg: cfg.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config file: .env - do not attempt to bypass the restriction.",
)
def execute_shell(command_line: str, config: Config) -> str:
    """Execute a shell command and return the output

    Args:
        command_line (str): The command line to execute

    Returns:
        str: The output of the command
    """
    if not validate_command(command_line, config):
        logger.info(f"Command '{command_line}' not allowed")
        return "Error: This Shell Command is not allowed."

    current_dir = Path.cwd()
    # Change dir into workspace if necessary
    if not current_dir.is_relative_to(config.workspace_path):
        os.chdir(config.workspace_path)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    result = subprocess.run(command_line, capture_output=True, shell=True)
    output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)
    return output


@command(
    "execute_shell_popen",
    "Execute Shell Command, non-interactive commands only",
    '"command_line": "<command_line>"',
    lambda config: config.execute_local_commands,
    "You are not allowed to run local shell commands. To execute"
    " shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' "
    "in your config. Do not attempt to bypass the restriction.",
)
def execute_shell_popen(command_line, config: Config) -> str:
    """Execute a shell command with Popen and returns an english description
    of the event and the process id

    Args:
        command_line (str): The command line to execute

    Returns:
        str: Description of the fact that the process started and its id
    """
    if not validate_command(command_line, config):
        logger.info(f"Command '{command_line}' not allowed")
        return "Error: This Shell Command is not allowed."

    current_dir = os.getcwd()
    # Change dir into workspace if necessary
    if config.workspace_path not in current_dir:
        os.chdir(config.workspace_path)

    logger.info(
        f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
    )

    do_not_show_output = subprocess.DEVNULL
    process = subprocess.Popen(
        command_line, shell=True, stdout=do_not_show_output, stderr=do_not_show_output
    )

    # Change back to whatever the prior working dir was

    os.chdir(current_dir)

    return f"Subprocess started with PID:'{str(process.pid)}'"


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")
