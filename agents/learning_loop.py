"""
Part 2: Learning from Doctor Edits
===================================
Implements:
1. A simulated "doctor reviewer" that applies a consistent editing policy
2. A reward signal based on edit distance
3. A correction-memory learning mechanism
4. Before/after metric reporting with improvement curve
"""

import json
import os
import time
import random
import math
from pathlib import Path
from typing import Optional
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ── Simulated Doctor Reviewer ─────────────────────────────────────────────────

DOCTOR_EDITING_POLICY = """
You are a senior clinician reviewing a discharge summary draft. You apply a consistent editing policy:

1. STYLE: Rewrite verbose or passive phrases to be concise and active.
2. TERMINOLOGY: Replace lay terms with proper clinical terminology.
3. STRUCTURE: Ensure hospital course follows: reason for admission → investigations → treatment → response → discharge.
4. COMPLETENESS: If a section says MISSING but the information was clearly available in context, note it.
5. MEDICATIONS: Ensure medications list is complete with dose/frequency/duration.
6. FOLLOW-UP: Make follow-up instructions specific (e.g., "OPD review in 1 week" not "follow up with doctor").
7. PENDING RESULTS: Ensure all pending results have a clear instruction (e.g., "Patient to collect urine C&S result and bring to follow-up").

Apply ONLY these policy-based edits. Do not add clinical information not in the original.
Return the edited discharge summary text.
"""


def simulate_doctor_edit(draft_text: str, client: OpenAI) -> dict:
    """
    Simulate a doctor editing the draft.
    Returns: { original, edited, edit_distance, reward }
    """
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": DOCTOR_EDITING_POLICY},
                {"role": "user", "content": f"Please edit this discharge summary draft:\n\n{draft_text}"},
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        edited = resp.choices[0].message.content.strip()
        dist = normalized_edit_distance(draft_text, edited)
        reward = 1.0 - dist  # Higher reward = less editing needed
        return {
            "original": draft_text,
            "edited": edited,
            "edit_distance": dist,
            "reward": reward,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Edit Distance ─────────────────────────────────────────────────────────────

def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein edit distance between two strings.
    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    # Work at word level for clinical text
    w1 = s1.lower().split()
    w2 = s2.lower().split()

    if not w1 and not w2:
        return 0.0
    if not w1 or not w2:
        return 1.0

    # Limit size to avoid memory issues
    w1 = w1[:500]
    w2 = w2[:500]

    m, n = len(w1), len(w2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if w1[i - 1] == w2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / max(m, n)


def section_accuracy(draft: str, edited: str) -> dict:
    """
    Compute per-section accuracy between draft and edited version.
    """
    sections = [
        "PATIENT INFORMATION",
        "DIAGNOSES",
        "HOSPITAL COURSE",
        "DISCHARGE MEDICATIONS",
        "FOLLOW-UP",
        "PENDING RESULTS",
        "FLAGS FOR REVIEW",
    ]
    scores = {}
    for section in sections:
        # Extract section text from both
        draft_section = _extract_section(draft, section)
        edited_section = _extract_section(edited, section)
        if draft_section and edited_section:
            dist = normalized_edit_distance(draft_section, edited_section)
            scores[section] = round(1.0 - dist, 3)
        elif not draft_section and not edited_section:
            scores[section] = 1.0
        else:
            scores[section] = 0.0
    return scores


def _extract_section(text: str, section_name: str) -> str:
    """Extract a section from the discharge summary."""
    lines = text.split("\n")
    in_section = False
    section_lines = []
    for line in lines:
        if section_name in line.upper():
            in_section = True
            continue
        if in_section:
            if line.strip() and any(
                kw in line.upper()
                for kw in ["PATIENT INFO", "DIAGNOS", "ALLERG", "HOSPITAL COURSE",
                           "DISCHARGE MED", "FOLLOW-UP", "PENDING", "FLAGS", "======"]
            ) and section_name not in line.upper():
                break
            section_lines.append(line)
    return " ".join(section_lines).strip()


# ── Correction Memory ─────────────────────────────────────────────────────────

class CorrectionMemory:
    """
    Stores (draft, edited) pairs and extracts recurring correction patterns.
    These are injected as few-shot examples into future prompts.
    """

    def __init__(self, memory_file: str = "output/correction_memory.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.pairs: list[dict] = []
        self.patterns: list[str] = []
        self._load()

    def _load(self):
        if self.memory_file.exists():
            with open(self.memory_file) as f:
                data = json.load(f)
                self.pairs = data.get("pairs", [])
                self.patterns = data.get("patterns", [])

    def save(self):
        with open(self.memory_file, "w") as f:
            json.dump({"pairs": self.pairs, "patterns": self.patterns}, f, indent=2)

    def add_pair(self, draft: str, edited: str, reward: float):
        self.pairs.append({
            "draft_snippet": draft[:500],
            "edited_snippet": edited[:500],
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep only last 20 pairs
        if len(self.pairs) > 20:
            self.pairs = self.pairs[-20:]
        self.save()

    def extract_patterns(self, client: OpenAI) -> list[str]:
        """Ask LLM to identify recurring correction patterns."""
        if len(self.pairs) < 2:
            return []

        pairs_text = "\n\n".join([
            f"Draft: {p['draft_snippet'][:200]}\nEdited: {p['edited_snippet'][:200]}"
            for p in self.pairs[-5:]
        ])

        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You identify recurring editing patterns in medical document corrections. Be concise."},
                    {"role": "user", "content": f"From these draft→edited pairs, list 3-5 recurring correction patterns the doctor consistently makes:\n\n{pairs_text}\n\nReturn as JSON list of strings."},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:]).rstrip("```")
            patterns = json.loads(content)
            self.patterns = patterns
            self.save()
            return patterns
        except Exception:
            return self.patterns


def build_memory_enhanced_prompt(base_prompt: str, memory: CorrectionMemory) -> str:
    """Inject correction patterns into a prompt to improve future outputs."""
    if not memory.patterns:
        return base_prompt

    pattern_block = "\n".join(f"  - {p}" for p in memory.patterns)
    injection = f"""
## LESSONS FROM PREVIOUS CLINICIAN EDITS
Based on past doctor corrections, you should pay special attention to:
{pattern_block}

Apply these patterns proactively in your output.
"""
    return base_prompt + "\n" + injection


# ── Learning Loop ─────────────────────────────────────────────────────────────

class LearningLoop:
    """
    Runs multiple iterations of:
    1. Generate draft
    2. Simulate doctor edit
    3. Compute reward
    4. Store in memory
    5. Extract patterns
    6. Improve next draft using patterns

    Reports before/after metrics and improvement curve.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory = CorrectionMemory(str(self.output_dir / "correction_memory.json"))

        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
            raise ValueError("DEEPSEEK_API_KEY not set in .env")

        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.results: list[dict] = []

    def run_iteration(self, draft_text: str, iteration: int) -> dict:
        """Run one learning iteration."""
        print(f"\n{'='*50}")
        print(f"  Learning Iteration {iteration}")
        print(f"{'='*50}")

        # Step 1: Simulate doctor edit
        print(f"  ▶ Simulating doctor review...")
        edit_result = simulate_doctor_edit(draft_text, self.client)
        if not edit_result["success"]:
            print(f"  ✗ Edit simulation failed: {edit_result['error']}")
            return {}

        reward = edit_result["reward"]
        edit_dist = edit_result["edit_distance"]
        section_scores = section_accuracy(draft_text, edit_result["edited"])

        print(f"  ✓ Edit distance: {edit_dist:.3f}")
        print(f"  ✓ Reward: {reward:.3f} (1.0 = no edits needed)")
        print(f"  ✓ Section scores: {json.dumps(section_scores, indent=4)}")

        # Step 2: Store in memory
        self.memory.add_pair(draft_text, edit_result["edited"], reward)

        # Step 3: Extract patterns
        patterns = self.memory.extract_patterns(self.client)
        if patterns:
            print(f"  ✓ Learned {len(patterns)} correction patterns")
            for p in patterns:
                print(f"    • {p}")

        result = {
            "iteration": iteration,
            "edit_distance": edit_dist,
            "reward": reward,
            "section_scores": section_scores,
            "patterns_learned": len(patterns),
        }
        self.results.append(result)
        return result

    def generate_improvement_report(self) -> str:
        """Generate before/after improvement report."""
        if len(self.results) < 2:
            return "Not enough iterations to report improvement."

        first = self.results[0]
        last = self.results[-1]

        improvement = first["edit_distance"] - last["edit_distance"]
        reward_improvement = last["reward"] - first["reward"]

        lines = [
            "\n" + "=" * 60,
            "  PART 2: LEARNING LOOP — IMPROVEMENT REPORT",
            "=" * 60,
            f"  Iterations run: {len(self.results)}",
            f"",
            f"  BEFORE (Iteration 1):",
            f"    Edit distance:  {first['edit_distance']:.3f}",
            f"    Reward:         {first['reward']:.3f}",
            f"",
            f"  AFTER (Iteration {len(self.results)}):",
            f"    Edit distance:  {last['edit_distance']:.3f}",
            f"    Reward:         {last['reward']:.3f}",
            f"",
            f"  IMPROVEMENT:",
            f"    Edit distance Δ: {improvement:+.3f} {'✓ IMPROVED' if improvement > 0 else '✗ DEGRADED'}",
            f"    Reward Δ:        {reward_improvement:+.3f}",
            f"",
            f"  ITERATION-BY-ITERATION CURVE:",
        ]
        for r in self.results:
            bar = "█" * int(r["reward"] * 20)
            lines.append(f"    Iter {r['iteration']:2d}: reward={r['reward']:.3f} {bar}")

        lines.append("")
        lines.append("  SECTION ACCURACY (Final Iteration):")
        for section, score in last.get("section_scores", {}).items():
            bar = "█" * int(score * 20)
            lines.append(f"    {section:<30s}: {score:.3f} {bar}")

        lines.append("")
        lines.append("  LIMITATIONS & SAFETY DISCUSSION:")
        lines.append("    1. Cold-start: With 1 patient, patterns may be overfitted to this")
        lines.append("       specific case. Real deployment needs ≥50 patient pairs.")
        lines.append("    2. Gaming risk: Agent could reduce edit distance by being vaguer.")
        lines.append("       Mitigation: reward must incorporate clinical completeness score,")
        lines.append("       not just edit distance alone.")
        lines.append("    3. Safety invariants: No-fabrication and escalation rules are")
        lines.append("       hardcoded in the agent core — they cannot be learned away.")
        lines.append("    4. Reviewer consistency: Simulated doctor applies a fixed policy;")
        lines.append("       real doctors disagree. Multi-annotator agreement needed in prod.")
        lines.append("=" * 60)

        report = "\n".join(lines)

        # Save report
        report_path = self.output_dir / "part2_learning_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved → {report_path}")

        # Save detailed JSON results
        json_path = self.output_dir / "part2_results.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)

        return report
