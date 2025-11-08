from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os, sys, re

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not set in .env")
    sys.exit(1)

client = Groq(api_key=api_key)

PREFERRED_PATTERNS = [
    r"llama[-_]3\.2[-_]?70b.*",
    r"llama[-_]3\.2[-_]?8b.*",
    r"llama[-_]3\.1[-_]?70b.*",
    r"llama[-_]3\.1[-_]?8b.*",
    r"mixtral.*",
]

SYSTEM_PROMPT = (
    "You are a helpful chef. Return a neat recipe with two sections: "
    "'Ingredients' (bulleted) and 'Steps' (numbered). Keep it under 200 words."
)

def pick_available_model() -> str:
    models = [m.id for m in client.models.list().data]
    for pat in PREFERRED_PATTERNS:
        for mid in models:
            if re.fullmatch(pat, mid, flags=re.IGNORECASE):
                return mid
    for mid in models:
        if "llama" in mid.lower():
            return mid
    if models:
        return models[0]
    raise SystemExit("❌ No models available on your Groq account.")

def chat_once(model: str, query: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Write a clear recipe for: {query}"},
        ],
        temperature=0.6,
        max_tokens=600,
    )
    msg = resp.choices[0].message
    if hasattr(msg, "content"):
        return (msg.content or "").strip()
    if isinstance(msg, dict):
        return (msg.get("content", "")).strip()
    return str(msg).strip()

def generate_recipe(query: str) -> str:
    last_err = None
    for m in [pick_available_model()]:
        try:
            print(f"→ Using model: {m}")
            return chat_once(m, query)
        except Exception as e:
            print(f"  ⚠️ {m} failed: {e}")
            last_err = e
    raise SystemExit(f"❌ No candidate models worked. Last error: {last_err}")

def main():
    dish = input("Enter a dish or ingredients: ").strip()
    if not dish:
        print("Please type something like: pizza margherita")
        return

    print("\n✅ Generating recipe with Groq...\n")
    recipe = generate_recipe(dish)
    print(recipe)

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    safe = (dish or "recipe").replace(" ", "_").lower()
    (outdir / f"{safe}.md").write_text(f"# {dish}\n\n{recipe}\n", encoding="utf-8")
    print(f"\n✅ Saved: outputs/{safe}.md")

if __name__ == "__main__":
    main()
