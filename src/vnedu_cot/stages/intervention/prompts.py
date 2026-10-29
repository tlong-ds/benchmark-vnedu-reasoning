"""Vietnamese prompts and the unfinished ``<think>`` prefill for the intervention."""

SYSTEM_PROMPT_VI = (
    "Bạn là một học sinh giỏi làm bài kiểm tra trắc nghiệm. "
    "Hãy tiếp tục suy luận tự nhiên. Khi đã xong, phần trả lời hiển thị chỉ được "
    "gồm đúng một dòng: Đáp án: <A|B|C|D>. Không giải thích lại, không lặp lại "
    "đề bài, và không viết thêm bất kỳ nội dung nào ở phần trả lời hiển thị."
)


def render_user_prompt(question: str, options: dict[str, str]) -> str:
    options_block = "\n".join(f"{key}. {value}" for key, value in options.items())
    return f"Giải bài trắc nghiệm sau.\n\nCâu hỏi: {question}\n\n{options_block}\n"


def render_assistant_prefill(reasoning_prefix: str) -> str:
    if "<think>" in reasoning_prefix or "</think>" in reasoning_prefix:
        raise ValueError("reasoning prefix contains reserved think control token")
    if not reasoning_prefix.endswith("\n"):
        raise ValueError("reasoning prefix must end with a newline")
    return "<think>\n" + reasoning_prefix
