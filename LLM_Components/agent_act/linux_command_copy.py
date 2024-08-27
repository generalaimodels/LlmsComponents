import re
import subprocess
from typing import Optional, List, Tuple
from g4f.client import Client

# Setting the event loop policy for Windows platforms if necessary
import sys
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

    # Few-shot examples to guide the model
    few_shot_examples: List[Tuple[str, str]] = [
        ("List all files in the current directory", "ls -la"),
        ("Find all Python files in the home directory", "find ~ -name '*.py'"),
        ("Show system uptime", "uptime"),
    ]

    # Constructing the prompt with detailed instructions
    prompt_segments: List[str] = [
        "Generate a Linux command based on the following description. Here are some examples:\n"
    ]
    
    for example, command in few_shot_examples:
        prompt_segments.append(
            f"Description: {example} \nCommand: {command}\n"
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
        # Clean up the generated text to ensure only the command is extracted
        command = generated_text.strip()
        # Remove unwanted prefixes like "Command:" and any backticks
        command = re.sub(r"^Command:\s*", "", command) # Remove prefix
        command = command.strip("`")  # Remove wrapping backticks if presen

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
        user_input = input("\nEnter command description: ").strip()

        if user_input.lower() == 'quit':
            print("Thank you for using the Linux Command Generator & Executor. Goodbye!")
            break

        if not user_input:
            print("Please enter a valid command description.")
            continue

        generated_command = generate_linux_command(user_input)
        
        if generated_command:
            print(f"{generated_command}")
            execute_confirmation = input("\nDo you want to execute this command? (yes/no): ").strip().lower()
            if execute_confirmation in ('yes', 'y'):
                execute_command(generated_command)
            else:
                print("Command execution skipped.")
        else:
            print("Unable to generate a command. Please try rephrasing your input or try again later.")


if __name__ == "__main__":
    main()