#%%
from IPython.display import Video, display
import os
# %%
def view_video(output_file):
    return Video(output_file, width=800, height=420)
# %%
for root, dirs, files in os.walk("data/filtering"):
    for f in files:
        if f.endswith(".mp4"):
            print(os.path.join(root, f))
            video = view_video(os.path.join(root, f))
            display(video)