from inspect_ai import task, Task, Epochs
from inspect_ai.solver import generate, system_message
from inspect_ai.model import GenerateConfig
from openbench.datasets.hle import get_dataset
from openbench.datasets.hle_250_ids import HLE_250_IDS
from openbench.scorers.hle import hle_scorer


# HLE system prompt as used in the original implementation
HLE_SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


@task
def hle(
    grader_model: str = "openai/o3-mini-2025-01-31", max_tokens: int = 8192
) -> Task:
    """Humanity's Last Exam: A benchmark at the frontier of human knowledge.

    HLE consists of 2,500 questions across dozens of subjects including mathematics,
    humanities, and natural sciences. Questions are designed by subject-matter experts
    globally and include both multiple-choice and short-answer formats.

    Args:
        grader_model: Model to use for grading responses (defaults to o3-mini-2025-01-31)
        max_tokens: Maximum tokens for model response (defaults to 8192 as recommended by HLE)

    Returns:
        Task configured for HLE evaluation
    """
    return Task(
        dataset=get_dataset(text_only=False),
        solver=[
            system_message(HLE_SYSTEM_PROMPT),
            generate(),
        ],
        scorer=hle_scorer(model=grader_model),
        name="hle",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation as per HLE
            max_tokens=max_tokens,  # HLE recommends at least 8192 for reasoning models
        ),
    )


@task
def hle_text(
    grader_model: str = "openai/o3-mini-2025-01-31", max_tokens: int = 8192
) -> Task:
    """Humanity's Last Exam (Text-Only): HLE with multi-modal questions filtered out.

    This variant includes only text-based questions from HLE, excluding any questions
    that require image understanding. Useful for evaluating models without vision capabilities.

    Args:
        grader_model: Model to use for grading responses (defaults to o3-mini-2025-01-31)
        max_tokens: Maximum tokens for model response (defaults to 8192 as recommended by HLE)

    Returns:
        Task configured for HLE text-only evaluation
    """
    return Task(
        dataset=get_dataset(text_only=True),
        solver=[
            system_message(HLE_SYSTEM_PROMPT),
            generate(),
        ],
        scorer=hle_scorer(model=grader_model),
        name="hle_text",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation as per HLE
            max_tokens=max_tokens,  # HLE recommends at least 8192 for reasoning models
        ),
    )


@task
def hle_250(
    grader_model: str = "openrouter/openai/gpt-5.6-luna",
    grader_reasoning_effort: str = "medium",
    max_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> Task:
    """Humanity's Last Exam (250): a fixed 250-question text-only HLE subset.

    Mirrors the shape of Artificial Analysis's Endpoint Accuracy Index HLE-250
    (their exact subset is private): 250 text-only questions stratified over the
    HLE categories, 10 epochs averaged, sampled at temperature=1.0/top_p=0.95,
    graded by an equality-checker judge. The question ids live in
    hle_250_ids.py; the judge defaults to GPT-5.6 Luna (medium) via OpenRouter,
    which needs OPENROUTER_API_KEY.

    Args:
        grader_model: Model to use for grading responses
        grader_reasoning_effort: Reasoning effort for the grader model
        max_tokens: Maximum tokens for model response
        temperature: Sampling temperature (non-zero, averaged over epochs)
        top_p: Nucleus sampling bound

    Returns:
        Task configured for HLE-250 evaluation
    """
    return Task(
        dataset=get_dataset(text_only=True, ids=HLE_250_IDS, name="hle_250"),
        solver=[
            system_message(HLE_SYSTEM_PROMPT),
            generate(),
        ],
        epochs=Epochs(10, "mean"),
        scorer=hle_scorer(
            model=grader_model, reasoning_effort=grader_reasoning_effort
        ),
        name="hle_250",
        config=GenerateConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
    )
