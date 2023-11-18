from dataclasses import dataclass

from chat.chat_result import ChatResult


@dataclass
class EvaluationResult:
    input_prompt: str
    score: str  # Score as a string
    issues: str  # Description of issues as a string
    reasoning: str  # Reasoning as a string
    final_matrices: str  # Final matrices information as a string
    causes_of_error: str  # Causes of error as a string
    proposed_fixes: str  # Proposed fixes as a string
    response: ChatResult # raw chat response from ChatGPT
