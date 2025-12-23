#%% pakete
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from pprint import pprint

#%% Output Models (Pydantic)
class YouTubeSuggestion(BaseModel):
    query: str = Field(description="YouTube search query (or channel + topic) to find relevant tutorials")
    why: str = Field(description="Short reason why this suggestion is helpful")
    duration_hint: str = Field(description="Suggested watch time, e.g. '10–15 min'")

class DayPlan(BaseModel):
    day: int = Field(description="Day number (1-7)")
    theme: str = Field(description="Theme of the day (e.g., rhythm, chord logic, improvisation)")
    warmup_minutes: int = Field(description="Warm-up duration in minutes (10-15 typical)")
    warmup_tasks: list[str] = Field(description="Concrete warm-up tasks (tempo/focus included if possible)")

    core_minutes: int = Field(description="Core concept duration in minutes (20-30 typical)")
    core_concept: str = Field(description="Clear explanation of the musical principle (no heavy theory)")

    practice_minutes: int = Field(description="Practice duration in minutes (20-30 typical)")
    practice_tasks: list[str] = Field(description="Concrete playing tasks, optionally with difficulty levels")

    song_minutes: int = Field(description="Song application duration in minutes (10-20 typical)")
    song_application: list[str] = Field(description="1-2 song ideas + what to listen for / play")

    youtube: list[YouTubeSuggestion] = Field(description="2-3 YouTube suggestions (queries), with reasons")

    reflection_minutes: int = Field(description="Reflection duration in minutes (about 5)")
    reflection_questions: list[str] = Field(description="2-3 targeted reflection questions")

    bonus_optional: list[str] = Field(default_factory=list, description="Optional: jam ideas, +15min variant, free application tips")

class GuitarWeekPlan(BaseModel):
    student_profile: str = Field(description="Short summary of the student context used for the plan")
    total_days: int = Field(description="Must be 7")
    days: list[DayPlan] = Field(description="Day-by-day plan (7 items)")

output_parser = PydanticOutputParser(pydantic_object=GuitarWeekPlan)

#%% Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
Du bist mein persönlicher Gitarrentutor. Erstelle einen 7-Tage-Gitarren-Intensivplan.
Wichtig:
- Praxisnah, klar, ohne trockenes Theoriewirrwarr.
- Kein Notenlesen voraussetzen.
- Jede Einheit soll spürbaren Fortschritt bringen.
- YouTube-Teil: Gib 2–3 konkrete Suchanfragen (oder Kanal + Thema), KEINE Links.
- Halte dich strikt an das JSON-Schema aus den Format-Instructions.

{format_instructions}
"""),
    ("user", """
Schülerprofil:
- Level: {level}
- Stil-Ziele: {styles}
- Gitarre: {guitar_type}
- Fokus: {focus}
- Zeit pro Tag: {minutes_per_day} Minuten
- Schwierigkeiten: {difficulties}

Erstelle jetzt den kompletten 7-Tage-Plan.
""")
]).partial(format_instructions=output_parser.get_format_instructions())

#%% Modellinstanz
MODEL_NAME = "llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME, temperature=0.4)

#%% Chain erstellen
chain = prompt_template | model | output_parser

#%% Chain Invocation (Beispielwerte)
res = chain.invoke({
    "level": "Anfänger bis leicht Fortgeschritten (ohne formale Theorie)",
    "styles": "Pop, Rock, bisschen Jazz; ich will auch jammen können",
    "guitar_type": "egal (akustisch oder elektrisch)",
    "focus": "Rhythmus, Akkordlogik (Stufen), saubere Wechsel, einfache Impro",
    "minutes_per_day": 60,
    "difficulties": "Timing schwankt, Barré anstrengend, ich verliere beim Wechsel den Groove"
})

#%% Ergebnis anzeigen
pprint(res.model_dump())

#%% Schönes Print pro Tag (optional)
print("\n" + "="*60)
print(res.student_profile)
print("="*60)

for d in res.days:
    print(f"\nTag {d.day}: {d.theme}")
    print("-"*60)
    print(f"Warm-up ({d.warmup_minutes} min):")
    for t in d.warmup_tasks:
        print(f"  - {t}")

    print(f"\nKernkonzept ({d.core_minutes} min):")
    print(f"  {d.core_concept}")

    print(f"\nPraxis ({d.practice_minutes} min):")
    for t in d.practice_tasks:
        print(f"  - {t}")

    print(f"\nSong-Anwendung ({d.song_minutes} min):")
    for s in d.song_application:
        print(f"  - {s}")

    print("\nYouTube (Suchanfragen):")
    for y in d.youtube:
        print(f"  - Query: {y.query}")
        print(f"    Warum: {y.why} | Watch: {y.duration_hint}")

    print(f"\nMini-Reflexion ({d.reflection_minutes} min):")
    for q in d.reflection_questions:
        print(f"  - {q}")

    if d.bonus_optional:
        print("\nBonus (optional):")
        for b in d.bonus_optional:
            print(f"  - {b}")
