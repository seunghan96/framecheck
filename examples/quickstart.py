"""FrameCheck in a few lines: check a drafted sentence and edit it by rule, with no model call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from framecheck import FrameCheck  # noqa: E402

fc = FrameCheck()

# Input facts given to the LLM: revenue rose 3%, on higher prices
facts = "Revenue: +3%. Higher prices."
draft = "Revenue surged 3% on higher prices, due to strong demand."

# Step 1: the prompt with six size-matched 10-K sentences of other firms
print(fc.build_prompt("Revenue", 2020, 3.0))
print("-" * 60)

# Step 2: check each judgment word against its usage norm and each cause against the input
report = fc.check(draft, size=3.0, context=facts)
for w in report["words"]:
    print(f"word  {w['lemma']:<10} class={w['cls']:<8} norm={w['range'][0]:g}-{w['range'][2]:g}%  flag={w['flag']}")
for c in report["causes"]:
    print(f"cause '{c['span']}'  support={c['support']}  supported={c['supported']}")
print("-" * 60)

# Step 3: edit only the flagged spans by rule, keeping the number
print("Draft:     ", draft)
print("FrameCheck:", fc.revise(draft, size=3.0, context=facts))

# A judgment word that fits the size is kept
print("Kept:      ", fc.revise("Revenue increased significantly by 45%.", size=45.0))

# A cause that the input gives is kept
print("Supported: ", fc.revise("Revenue increased 3%, driven by higher prices.", size=3.0, context=facts))
