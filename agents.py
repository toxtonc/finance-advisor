"""
agents.py — ExtractionAgent, QueryAgent, AdvisorAgent.

Each agent is a simple class with a run() method.
ExtractionAgent and QueryAgent return parsed Python objects (blocking).
AdvisorAgent is a generator that streams tokens.
"""

import json
import re
import ollama


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


class ExtractionAgent:
    """
    Parses a free-text financial situation into a structured profile dict.
    Model: qwen3.5:2b — constrained extraction task.
    """

    MODEL = "qwen3.5:2b"
    SYSTEM_PROMPT = (
        "DO NOT THINK. SKIP REASONING. RESPOND DIRECTLY WITH JSON.\n"
        "You are a financial data extraction assistant. "
        "Read the user's description of their financial situation and extract a structured profile. "
        "Rules:\n"
        "- Extract ONLY information explicitly stated or clearly implied in the text.\n"
        "- Use the string \"unknown\" for any field not mentioned.\n"
        "- Do NOT invent, assume, or estimate numbers not present in the input.\n"
        "- Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown.\n"
        "The JSON object must have exactly these keys:\n"
        "  monthly_income, monthly_expenses, savings, debts, goals, "
        "risk_tolerance, time_horizon, summary\n"
        "risk_tolerance must be one of: \"low\", \"medium\", \"high\", \"unknown\".\n"
        "summary must be a single sentence in plain English summarising the profile."
    )

    def run(self, user_text: str) -> dict:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        try:
            response = ollama.chat(
                model=self.MODEL,
                messages=messages,
                format="json",
                think=False,
                options={"temperature": 0.1},
            )
            raw = response.message.content
        except Exception as e:
            raise RuntimeError(f"ExtractionAgent Ollama call failed: {e}") from e

        if not raw.strip():
            raise RuntimeError("ExtractionAgent returned an empty response.")

        cleaned = _strip_fences(raw)
        try:
            profile = json.loads(cleaned)
        except json.JSONDecodeError:
            # Retry once with an explicit fix request
            try:
                fix_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Your response was not valid JSON. "
                     "Reply ONLY with the corrected JSON object, nothing else."},
                ]
                response2 = ollama.chat(
                    model=self.MODEL,
                    messages=fix_messages,
                    format="json",
                    think=False,
                    options={"temperature": 0.1},
                )
                cleaned2 = _strip_fences(response2.message.content)
                profile = json.loads(cleaned2)
            except Exception as e2:
                # If everything fails, raise a detailed error with the raw content
                raise RuntimeError(
                    f"ExtractionAgent failed to produce valid JSON after retry. "
                    f"Raw output from model: '{raw}' | Parse Error: {e2}"
                ) from e2

        return profile


class QueryAgent:
    """
    Translates a financial profile dict into 3 targeted retrieval queries.
    Model: qwen3.5:2b — semantic inference task.
    """

    MODEL = "qwen3.5:2b"
    SYSTEM_PROMPT = (
        "DO NOT THINK. SKIP REASONING. RESPOND DIRECTLY WITH JSON ARRAY.\n"
        "You are a retrieval query generation assistant for a personal finance RAG system. "
        "Given a structured financial profile, generate exactly 3 retrieval queries that together "
        "maximise the diversity and coverage of the knowledge base search.\n"
        "Rules:\n"
        "- Each query MUST cover a DIFFERENT financial dimension. Assign one query to each of "
        "these three slots:\n"
        "  SLOT A — immediate situation. PRIORITY RULE: if the 'debts' field is non-zero and "
        "not 'unknown', this slot MUST target debt-repayment strategy (e.g. high-interest debt, "
        "credit card balances). Only if the user has no debts, target their cash-flow or "
        "expense-management problem instead.\n"
        "  SLOT B — savings or investment strategy: address how to grow or protect capital given "
        "the user's risk tolerance and time horizon.\n"
        "  SLOT C — long-term goal: address the specific financial goal(s) stated in the profile "
        "(e.g. retirement, home purchase, education fund, emergency fund).\n"
        "- No two queries may share the same root topic or repeat the same keywords.\n"
        "- Do NOT copy, paraphrase, or reproduce phrases from the 'summary' or 'goals' fields "
        "of the profile. Those fields describe the situation in generic terms; your queries must "
        "be specific, actionable retrieval phrases that go beyond them.\n"
        "- Each query must be 5 to 12 words long.\n"
        "- Queries MUST be phrased as topical, information-seeking noun phrases — NOT questions, "
        "NOT imperative sentences, NOT advice statements.\n"
        "- Do NOT start a query with an imperative verb (e.g. 'Reduce', 'Build', 'Invest', "
        "'Pay', 'Save'). Use noun-phrase phrasing instead (e.g. 'strategies for paying off "
        "credit card debt', 'low-risk short-term savings vehicles', 'planning for a car "
        "purchase on a tight budget').\n"
        "- Do NOT include any specific numbers from the profile (amounts, durations, "
        "percentages, currencies). Queries must be concept-level, not numeric.\n"
        "- Do NOT use question words (what, how, why, when, etc.).\n"
        "- Respond ONLY with a valid JSON array of exactly 3 strings in order [A, B, C]. "
        "No preamble, no explanation."
    )

    def run(self, profile: dict) -> list[str]:
        profile_text = "\n".join(f"{k}: {v}" for k, v in profile.items())
        forbidden = (
            f"Forbidden phrases to avoid copying: "
            f"\"{profile.get('summary', '')}\" | \"{profile.get('goals', '')}\""
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Financial profile:\n{profile_text}\n\n{forbidden}"},
        ]

        fallback = [
            profile.get("summary", "personal finance advice"),
            profile.get("goals", "financial goals"),
            (profile.get("summary", "personal finance") + " " + profile.get("goals", "goals")).strip(),
        ]

        try:
            response = ollama.chat(
                model=self.MODEL,
                messages=messages,
                format="json",
                think=False,
                options={"temperature": 0.2},
            )
            raw = response.message.content
        except Exception:
            return fallback

        cleaned = _strip_fences(raw)
        try:
            queries = json.loads(cleaned)
            if not isinstance(queries, list) or len(queries) != 3:
                return fallback
            return [str(q) for q in queries]
        except json.JSONDecodeError:
            return fallback


class AdvisorAgent:
    """
    Generates personalised financial advice grounded in retrieved context chunks.
    Model: qwen3.5:4b — streaming generator.
    """

    MODEL = "qwen3.5:4b"
    SYSTEM_PROMPT = (
        "DO NOT THINK. SKIP REASONING. RESPOND DIRECTLY.\n"
        "CRITICAL: Do NOT use the '$' symbol for currency. Always use 'USD' or 'dollars' instead to prevent rendering issues.\n\n"
        "You are a helpful and honest personal finance advisor. "
        "Your advice must be grounded STRICTLY in the provided context excerpts. "
        "Do NOT invent specific numbers, interest rates, product names, or statistics "
        "that are not mentioned in the context.\n"
        "If the provided context does not contain relevant advice for the user's profile, "
        "you MUST NOT offer generic advice. Instead, say exactly: 'I do not have enough "
        "information based on the sources to provide detailed advice on this.'\n\n"
        "Structure your response with these sections using markdown. "
        "Each section title MUST be a level-2 markdown heading (start the line with '## '), "
        "NOT a numbered list item and NOT bold text. Use exactly these two headings, in this order:\n"
        "## Key Recommendations\n"
        "(3 to 5 bullet points of actionable advice)\n"
        "## Relevant Principles\n"
        "(briefly note which concepts from the sources apply)\n\n"
        "Do NOT include a profile summary or financial snapshot.\n"
        "Keep the total response under 400 words. Use markdown formatting (bold, bullet points, tables)."
    )

    def run(self, profile: dict, chunks: list[dict]):
        """Generator that yields string tokens from the streaming Ollama response."""
        profile_text = "\n".join(f"{k}: {v}" for k, v in profile.items())

        context_parts = []
        for chunk in chunks:
            context_parts.append(f"--- Source: {chunk['source']} ---\n{chunk['text']}")
        context_text = "\n\n".join(context_parts)

        user_content = (
            f"Financial profile:\n{profile_text}\n\n"
            f"Retrieved context:\n{context_text}"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            stream = ollama.chat(
                model=self.MODEL,
                messages=messages,
                stream=True,
                think=False,
                options={"temperature": 0.1},
            )
            for chunk in stream:
                yield chunk.message.content
        except Exception as e:
            yield f"\n\n[Error generating advice: {e}]"
