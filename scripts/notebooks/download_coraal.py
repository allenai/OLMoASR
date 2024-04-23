# %%
import subprocess
import os

# %%
with open("/home/ubuntu/open_whisper/data/eval/coraalfiles.txt", "r") as f:
    coraalfiles = [line.strip() for line in f]
# %%
coraalfiles[:10]
# %%
for f in coraalfiles:
    if "audio" in f:
        subprocess.run(
            ["wget", f, "-P", "/home/ubuntu/open_whisper/data/eval/coraal"]
        )
    
    if "textfiles" in f:
        subprocess.run(
            ["wget", f, "-P", "/home/ubuntu/open_whisper/data/eval/coraal"]
        )
    


# %%
