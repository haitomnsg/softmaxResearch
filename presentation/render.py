"""Render the deck with the real PowerPoint (best-fidelity QA on this machine)."""
import os, glob, sys
import win32com.client

deck = os.path.abspath("GAM-Softmax-Presentation.pptx")
out = os.path.abspath("render")
os.makedirs(out, exist_ok=True)
for f in glob.glob(os.path.join(out, "*.jpg")) + glob.glob(os.path.join(out, "*.png")):
    os.remove(f)

app = win32com.client.Dispatch("PowerPoint.Application")
pres = app.Presentations.Open(deck, WithWindow=False)
pres.Export(out, "JPG", 1400, 788)
pres.Close()
app.Quit()
print("slides:", len(glob.glob(os.path.join(out, "*.JPG")) + glob.glob(os.path.join(out, "*.jpg"))))
