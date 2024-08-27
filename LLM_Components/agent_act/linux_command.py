import re
import subprocess
from typing import Optional, List, Tuple
from g4f.client import Client
import sys
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows compatibility
colorama.init(autoreset=True)

# Setting the event loop policy for Windows platforms if necessary
if sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def generate_linux_command(input_str: str) -> Optional[str]:
    """
    Generate a Linux command based on the input string using advanced prompting techniques.
    
    Args:
        input_str (str): The user's input describing the desired Linux command.
    
    Returns:
        Optional[str]: The generated Linux command or None if the command cannot be generated.
    """
    client = Client()

    few_shot_examples: List[Tuple[str, str]] = [
        ("List all files in the current directory", "ls -la"),
        ("Find all Python files in the home directory", "find ~ -name '*.py'"),
        ("Show system uptime", "uptime"),
    ]

    prompt_segments: List[str] = [
        "Generate a Linux command based on the following description. Here are some examples:\n"
    ]
    
    for example, command in few_shot_examples:
        prompt_segments.append(
            f"Description: {example}\nCommand: {command}\n"
            f"Explanation: This command lists all files, searches Python files, or shows uptime.\n"
        )

    prompt_segments.append(
        f"Now, generate a command for this description: '{input_str}'\n"
        "Just provide the command text without any additional commentary, formatting, or prefixes.\n"
    )
    prompt = "\n".join(prompt_segments)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        generated_text = response.choices[0].message.content
        command = generated_text.strip()
        command = re.sub(r"^Command:\s*", "", command)
        command = command.strip("`")

        return command
    
    except Exception as e:
        print(f"Error generating command: {str(e)}")
    
    return None


def execute_command(command: str) -> None:
    """
    Execute the given Linux command and display the result/output.

    Args:
        command (str): The Linux command to execute.
    """
    print(f"\nExecuting Command: {command}\n")
    try:
        result = subprocess.run(command, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Command Output:\n")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Command Errors:\n")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error during command execution: {e.stderr.strip()}")


def main() -> None:
    """
    Main function to run the Linux command generator and executor.
    """
    print("Welcome to the Advanced Linux Command Generator & Executor!")
    print("Enter your command description or type 'quit' to exit.")

    while True:
        user_input = input(f"\n{Fore.YELLOW}Enter command description: {Style.RESET_ALL}").strip()

        if user_input.lower() == 'quit':
            print("Thank you for using the Linux Command Generator & Executor. Goodbye!")
            break

        if not user_input:
            print(f"{Fore.RED}Please enter a valid command description.{Style.RESET_ALL}")
            continue

        generated_command = generate_linux_command(user_input)
        
        if generated_command:
            print(f"{Fore.GREEN}{generated_command}{Style.RESET_ALL}")
            execute_confirmation = input(f"\n{Fore.BLUE}Do you want to execute this command? (yes/no): {Style.RESET_ALL}").strip().lower()
            if execute_confirmation in ('yes', 'y'):
                execute_command(generated_command)
            else:
                print("Command execution skipped.")
        else:
            print(f"{Fore.RED}Unable to generate a command. Please try rephrasing your input or try again later.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()