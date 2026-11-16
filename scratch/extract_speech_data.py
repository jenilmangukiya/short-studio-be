import json

with open("storage/e8342d48-2150-4b46-b05e-88c454f1fc87/subtitles.json") as f:
    d = json.load(f)

segments = d.get("segments", [])
print(f"Total segments: {len(segments)}")

all_words = []
for i, s in enumerate(segments):
    st = s.get("start") or 0.0
    et = s.get("end") or 0.0
    txt = s.get("text")
    words = s.get("words", [])
    print(f"\n--- Segment {i+1} ({st:.3f}s -> {et:.3f}s, dur: {et-st:.3f}s) ---")
    print(f"Text: \"{txt}\"")
    for w in words:
        all_words.append(w)
        w_dur = w['end'] - w['start']
        print(f"  {w['text']:<15} {w['start']:.3f}s -> {w['end']:.3f}s (dur: {w_dur:.3f}s)")

# Analyze pauses between words and between segments
print("\n" + "="*60)
print("INTER-WORD & INTER-SEGMENT PAUSES (> 40ms):")
print("="*60)
pauses = []
for i in range(len(all_words) - 1):
    w1 = all_words[i]
    w2 = all_words[i+1]
    gap = w2["start"] - w1["end"]
    if gap > 0.04:
        pauses.append((w1["text"], w2["text"], gap, w1["end"], w2["start"]))

for w1, w2, gap, t1, t2 in pauses:
    marker = " [SENTENCE/CLAUSE PAUSE]" if gap > 0.20 else ""
    if gap > 0.45:
        marker = " *** UNUSUALLY LONG PAUSE ***"
    print(f"Pause after '{w1}' before '{w2}': {gap:.3f}s ({t1:.3f}s -> {t2:.3f}s){marker}")

avg_pause = sum(p[2] for p in pauses) / len(pauses) if pauses else 0
print(f"\nTotal pauses (>40ms): {len(pauses)}")
print(f"Average pause duration: {avg_pause:.3f}s")

# Speech rate analysis per segment
print("\n" + "="*60)
print("SPEAKING RATE PER SEGMENT (Words Per Second):")
print("="*60)
for i, s in enumerate(segments):
    words = s.get("words", [])
    if not words:
        continue
    first_w = words[0]['start']
    last_w = words[-1]['end']
    speech_dur = last_w - first_w
    wps = len(words) / speech_dur if speech_dur > 0 else 0
    print(f"Segment {i+1} (\"{s.get('text')[:35]}...\"): {len(words)} words in {speech_dur:.2f}s -> {wps:.2f} words/sec")
