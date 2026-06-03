#!/usr/bin/env python3
"""
Discharge Summary Agent — Main Entry Point
==========================================
Usage:
  python main.py                          # Run on all patients in data/patients/
  python main.py --patient patient_2      # Run on specific patient
  python main.py --patient patient_2 --part2  # Also run Part 2 learning loop
  python main.py --demo                   # Run demo mode with synthetic patient

Environment:
  Copy .env.example to .env and add your DeepSeek API key.
"""

import argparse
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def check_env():
    """Check that required environment is set up."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key or api_key == "your_deepseek_api_key_here":
        console.print(Panel(
            "[bold red]❌ DeepSeek API Key Missing![/bold red]\n\n"
            "1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan]\n"
            "2. Open [cyan].env[/cyan] and replace [yellow]your_deepseek_api_key_here[/yellow] with your actual key\n"
            "3. Get your key at: [link=https://platform.deepseek.com]https://platform.deepseek.com[/link]\n\n"
            "Example:\n"
            "  [green]DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx[/green]",
            title="Setup Required",
            border_style="red",
            box=box.DOUBLE_EDGE,
        ))
        sys.exit(1)

    return api_key


def run_on_patient(patient_id: str, patient_folder: str, output_dir: str = "output"):
    """Run the discharge summary agent on a single patient."""
    from agents.discharge_agent import run_agent

    console.print(f"\n[bold cyan]Running agent for:[/bold cyan] {patient_id}")
    console.print(f"[dim]Folder:[/dim] {patient_folder}")
    console.print(f"[dim]Output:[/dim] {output_dir}\n")

    summary = run_agent(
        patient_folder=patient_folder,
        patient_id=patient_id,
        output_dir=output_dir,
    )

    console.print(f"\n[bold green]✅ Complete! Outputs saved to:[/bold green]")
    console.print(f"  📄 {output_dir}/{patient_id}_discharge_summary.txt")
    console.print(f"  📊 {output_dir}/{patient_id}_discharge_summary.json")
    console.print(f"  🔍 {output_dir}/{patient_id}_trace.json")

    return summary


def run_part2(summary_text: str, output_dir: str = "output", iterations: int = 3):
    """Run the Part 2 learning loop."""
    from agents.learning_loop import LearningLoop

    console.print(Panel(
        "[bold cyan]Part 2: Learning from Doctor Edits[/bold cyan]\n"
        f"Running {iterations} iterations of the simulated review loop...",
        border_style="cyan",
    ))

    loop = LearningLoop(output_dir=output_dir)

    for i in range(1, iterations + 1):
        loop.run_iteration(summary_text, iteration=i)

    report = loop.generate_improvement_report()
    console.print(report)

    return loop


def create_synthetic_patient_2(base_dir: str = "data/patients"):
    """
    Create a synthetic second patient folder for demo purposes.
    This represents a different patient with conflicting data and missing fields
    to showcase the agent's safety features.
    """
    patient_dir = Path(base_dir) / "patient_demo"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Write a synthetic text file as a "clinical note" — we'll process it specially
    note_path = patient_dir / "synthetic_note.txt"
    with open(note_path, "w") as f:
        f.write("""
ADMISSION NOTE
==============
Patient: [REDACTED FOR DEMO]
Date of Admission: 05/02/2026
Ward: General Medicine

Chief Complaints: Chest pain x 2 days, shortness of breath

History of Present Illness:
Patient presented with progressive chest pain and dyspnoea. 
History of hypertension and type 2 diabetes on oral hypoglycemics.

Vitals: BP 160/100 mmHg, PR 92/min, RR 22/min, SpO2 94% on room air, Temp 37.2°C

Allergies: NSAID allergy documented

Past History: Hypertension (10 years), Type 2 Diabetes Mellitus

PROGRESS NOTE (Day 3)
======================
Date: 08/02/2026
Patient improving. Repeat ECG shows no new changes.
Troponin: NEGATIVE (serial)
Echo: EF 55%, no wall motion abnormality

CONFLICTING DATA: 
- Admission note diagnosis: "Unstable Angina"
- Progress note Day 3 diagnosis: "Musculoskeletal chest pain"
These conflict and require clinician review.

DISCHARGE NOTE
==============
Date of Discharge: 10/02/2026
Condition: Stable
Discharge Medications:
1. Tab Metformin 500mg - 1-0-1 - Continue
2. Tab Amlodipine 5mg - 0-0-1 - Continue
3. Tab Aspirin 75mg - 1-0-0 - NEW (no documented indication — NOTE: patient has NSAID allergy history)
4. Tab Pantoprazole 40mg - 1-0-0 - NEW

Follow-up: OPD review in 2 weeks
Urine microalbumin: PENDING - report awaited
""")

    return str(patient_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Discharge Summary Agent — Clinical AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           Run on all patients in data/patients/
  python main.py --patient patient_2       Run on patient_2 only
  python main.py --patient patient_2 --part2  Also run learning loop
  python main.py --demo                    Run demo with 2 patients including conflict case
        """,
    )
    parser.add_argument("--patient", type=str, help="Specific patient ID to process")
    parser.add_argument("--patients-dir", type=str, default="data/patients", help="Directory containing patient folders")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--part2", action="store_true", help="Run Part 2 learning loop after generating summary")
    parser.add_argument("--part2-iterations", type=int, default=3, help="Number of learning loop iterations")
    parser.add_argument("--demo", action="store_true", help="Run full demo including synthetic conflict patient")
    args = parser.parse_args()

    # Header
    console.print(Panel(
        "[bold cyan]🏥 Discharge Summary Agent[/bold cyan]\n"
        "[dim]Agentic AI for Clinical Document Processing[/dim]\n\n"
        "Built for safety: Every unknown fact is flagged, never fabricated.",
        box=box.DOUBLE_EDGE,
        border_style="cyan",
    ))

    # Check API key
    check_env()

    last_summary = None

    if args.demo:
        # Run patient_2 first
        p2_folder = Path(args.patients_dir) / "patient_2"
        if p2_folder.exists():
            summary = run_on_patient("patient_2", str(p2_folder), args.output_dir)
            last_summary = summary

        # Run synthetic conflict patient
        console.print("\n[bold yellow]Creating synthetic conflict patient for demo...[/bold yellow]")
        demo_folder = create_synthetic_patient_2(args.patients_dir)
        summary = run_on_patient("patient_demo", demo_folder, args.output_dir)
        last_summary = summary

    elif args.patient:
        patient_folder = Path(args.patients_dir) / args.patient
        if not patient_folder.exists():
            console.print(f"[red]❌ Patient folder not found: {patient_folder}[/red]")
            sys.exit(1)
        summary = run_on_patient(args.patient, str(patient_folder), args.output_dir)
        last_summary = summary

    else:
        # Run on all patients
        patients_dir = Path(args.patients_dir)
        if not patients_dir.exists():
            console.print(f"[red]❌ Patients directory not found: {patients_dir}[/red]")
            sys.exit(1)

        patient_folders = [d for d in patients_dir.iterdir() if d.is_dir()]
        if not patient_folders:
            console.print(f"[red]❌ No patient folders found in {patients_dir}[/red]")
            sys.exit(1)

        console.print(f"[cyan]Found {len(patient_folders)} patient(s)[/cyan]")
        for folder in sorted(patient_folders):
            summary = run_on_patient(folder.name, str(folder), args.output_dir)
            last_summary = summary

    # Part 2 - Learning Loop — prefer patient_2 over patient_demo
    if (args.part2 or args.demo) and last_summary is not None:
        # Use patient_2 summary if it exists (more content = better learning signal)
        preferred_path = Path(args.output_dir) / "patient_2_discharge_summary.txt"
        fallback_path = Path(args.output_dir) / f"{last_summary.patient_id}_discharge_summary.txt"
        summary_path = preferred_path if preferred_path.exists() else fallback_path
        if summary_path.exists():
            with open(summary_path) as f:
                summary_text = f.read()
            run_part2(summary_text, args.output_dir, args.part2_iterations)
        else:
            console.print("[yellow]⚠ Could not find summary text for Part 2. Skipping.[/yellow]")

    console.print("\n[bold green]✨ All done![/bold green]")


if __name__ == "__main__":
    main()
