"""CoT / direct prompt templates for chart QA fine-tuning and eval."""

from __future__ import annotations

COT_PROMPT_TEMPLATE = """You are given a chart image and a question about it.

Answer using chain-of-thought reasoning, then give the final answer.

Your reply MUST use exactly two XML-style sections in this order:

<think>
Your step-by-step reasoning about the chart.
</think>

<answer>
The final answer only (concise; match the style expected for chart QA).
</answer>

Question:
{question}"""

DIRECT_PROMPT_TEMPLATE = """You are given a chart image and a question about it.

Answer the question directly and concisely from the chart only. Do not show reasoning or chain-of-thought.

Question:
{question}"""

COT_FORMAT_ICL_PROMPT_TEMPLATE = """You are given a chart image and a question about it.

Answer using chain-of-thought reasoning, then give the final answer.

Your reply MUST use exactly two XML-style sections in this order:

<think>
Your step-by-step reasoning about the chart.
</think>

<answer>
The final answer only (concise; match the style expected for chart QA).
</answer>

Here are two examples of the expected reply format (different charts; format only):

Example 1 — question: How many bars appear to be above the average height?
<think>
Compare each bar height to the mean; count those strictly above it.
</think>
<answer>
3
</answer>

Example 2 — question: Is the largest bar more than twice the size of the smallest bar?
<think>
Read the max and min values from the chart and check whether max > 2 * min.
</think>
<answer>
Yes
</answer>

Now answer the following question about the provided chart image.

Question:
{question}"""


def build_cot_prompt(question: str) -> str:
    return COT_PROMPT_TEMPLATE.format(question=(question or "").strip())


def build_direct_prompt(question: str) -> str:
    return DIRECT_PROMPT_TEMPLATE.format(question=(question or "").strip())


def build_cot_format_icl_prompt(question: str) -> str:
    return COT_FORMAT_ICL_PROMPT_TEMPLATE.format(question=(question or "").strip())
