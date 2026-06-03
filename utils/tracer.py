"""
Trace Logger - Observability for the agent loop.
Emits a readable step trace: reasoning → action → inputs → result → next decision.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()


class TraceStep:
    def __init__(self, step_num: int, reasoning: str):
        self.step_num = step_num
        self.reasoning = reasoning
        self.action: Optional[str] = None
        self.inputs: Optional[dict] = None
        self.result: Optional[Any] = None
        self.next_decision: Optional[str] = None
        self.timestamp = datetime.now().isoformat()
        self.duration_ms: Optional[float] = None
        self._start = time.time()

    def complete(self, action: str, inputs: dict, result: Any, next_decision: str):
        self.action = action
        self.inputs = inputs
        self.result = result
        self.next_decision = next_decision
        self.duration_ms = round((time.time() - self._start) * 1000, 1)

    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "reasoning": self.reasoning,
            "action": self.action,
            "inputs": self.inputs,
            "result": self.result,
            "next_decision": self.next_decision,
        }


class AgentTracer:
    def __init__(self, patient_id: str, output_dir: str = "output"):
        self.patient_id = patient_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps: list[TraceStep] = []
        self.warnings: list[str] = []
        self.escalations: list[dict] = []
        self.start_time = time.time()
        self._current_step: Optional[TraceStep] = None

        console.print(
            Panel(
                f"[bold cyan]🏥 Discharge Summary Agent[/bold cyan]\n"
                f"[dim]Patient: {patient_id} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
                box=box.DOUBLE_EDGE,
                border_style="cyan",
            )
        )

    def start_step(self, step_num: int, reasoning: str) -> TraceStep:
        step = TraceStep(step_num, reasoning)
        self._current_step = step
        self.steps.append(step)

        console.print(f"\n[bold yellow]━━━ Step {step_num} ━━━[/bold yellow]")
        console.print(f"[italic dim]🧠 Reasoning:[/italic dim] {reasoning}")
        return step

    def complete_step(self, action: str, inputs: dict, result: Any, next_decision: str):
        if self._current_step:
            self._current_step.complete(action, inputs, result, next_decision)

        # Format result for display
        result_str = str(result)
        if len(result_str) > 300:
            result_str = result_str[:300] + "... [truncated]"

        console.print(f"[green]⚡ Action:[/green] [bold]{action}[/bold]")
        if inputs:
            inputs_str = json.dumps(inputs, default=str)
            if len(inputs_str) > 200:
                inputs_str = inputs_str[:200] + "..."
            console.print(f"[dim]   Inputs: {inputs_str}[/dim]")
        console.print(f"[blue]📤 Result:[/blue] {result_str}")
        console.print(f"[magenta]➡ Next:[/magenta] {next_decision}")

    def log_warning(self, message: str):
        self.warnings.append(message)
        console.print(f"[bold red]⚠ WARNING:[/bold red] {message}")

    def log_escalation(self, reason: str, details: dict):
        entry = {"reason": reason, "details": details, "timestamp": datetime.now().isoformat()}
        self.escalations.append(entry)
        console.print(
            Panel(
                f"[bold red]🚨 ESCALATION FLAGGED[/bold red]\n{reason}",
                border_style="red",
            )
        )

    def log_info(self, message: str):
        console.print(f"[cyan]ℹ {message}[/cyan]")

    def log_success(self, message: str):
        console.print(f"[bold green]✓ {message}[/bold green]")

    def save_trace(self) -> str:
        """Save full trace as JSON."""
        trace_data = {
            "patient_id": self.patient_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration_seconds": round(time.time() - self.start_time, 2),
            "total_steps": len(self.steps),
            "warnings_count": len(self.warnings),
            "escalations_count": len(self.escalations),
            "warnings": self.warnings,
            "escalations": self.escalations,
            "steps": [s.to_dict() for s in self.steps],
        }

        trace_path = self.output_dir / f"{self.patient_id}_trace.json"
        with open(trace_path, "w") as f:
            json.dump(trace_data, f, indent=2, default=str)

        console.print(f"\n[dim]Trace saved → {trace_path}[/dim]")
        return str(trace_path)

    def print_summary(self):
        elapsed = round(time.time() - self.start_time, 1)
        console.print(
            Panel(
                f"[bold green]✅ Agent Complete[/bold green]\n"
                f"Steps: {len(self.steps)} | "
                f"Warnings: {len(self.warnings)} | "
                f"Escalations: {len(self.escalations)} | "
                f"Time: {elapsed}s",
                border_style="green",
            )
        )
