"""Shared prompt and request contracts for the tutorial's measured scenarios."""

DEFAULT_LENGTHS = [32, 128, 1024, 2048]
OUTPUT_TOKENS = 8
OFFICIAL_ONE_TOKEN = [3847, 284, 1373, 8282, 4218, 3755, 947, 906]
PROMPT_TEXT = "Explain why the sky is blue, and how sunlight interacts with air. "


def validate_lengths(lengths):
    if (not lengths or len(set(lengths)) != len(lengths)
            or any(type(length) is not int or length < 1 for length in lengths)
            or max(lengths) + OUTPUT_TOKENS > 4096):
        raise ValueError("Use distinct positive prompt lengths within the 4096-token cache")
    return lengths


def make_prompts(checkpoint, lengths):
    validate_lengths(lengths)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                             trust_remote_code=True)
    source = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    if not source:
        raise ValueError("The checkpoint tokenizer produced an empty prompt")
    return {length: ([11] if length == 1 else
                     (source * ((length + len(source) - 1) // len(source)))[:length])
            for length in lengths}


def report_scenarios(report):
    if "scenarios" in report:
        return report["scenarios"]
    return [dict(prompt_length=len(report["prompt"]), prompt=report["prompt"],
                 output_tokens=report["output_tokens"], requests=report["requests"])]


def reference_scenarios(report, lengths):
    validate_lengths(lengths)
    if not report["passed"] or report["versions"]["vllm"] != "0.31.0":
        raise ValueError("An independently completed vLLM 0.31.0 reference is required")
    scenarios = {row["prompt_length"]: row for row in report_scenarios(report)}
    if set(scenarios) != set(lengths):
        raise ValueError("Reference and candidate prompt lengths differ")
    for length, row in scenarios.items():
        requests = row["requests"]
        if len(row["prompt"]) != length or row["output_tokens"] != OUTPUT_TOKENS or len(requests) < 2:
            raise ValueError("Incomplete independent reference requests")
        tokens = requests[0]["tokens"]
        if (len(tokens) != OUTPUT_TOKENS or any(type(t) is not int or t < 0 for t in tokens)
                or any(request["tokens"] != tokens for request in requests)):
            raise ValueError("Independent native greedy requests disagree")
    return scenarios
