# %%
import ray
import glob
from open_whisper.utils import TranscriptReader
import json
from typing import Optional, Literal
import numpy as np
import os
import glob
from open_whisper.utils import TranscriptReader
import json
from typing import Optional, Literal
import numpy as np
import os
# %%
def bytes_to_text(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
    transcript_dict["text"] = transcript_dict["bytes"].decode("utf-8")
    del transcript_dict["bytes"]
    return transcript_dict


ds = ray.data.read_binary_files(
    paths="data/00000", file_extensions=["srt"], include_paths=True
).map(bytes_to_text)
# %%
ds.show()
# %%
ds.count()

# %%
def check_case(transcript_dict: Dict[str, Any]) -> Dict[str, Any]:
    reader = TranscriptReader(transcript_string=transcript_dict["text"], ext="srt")
    t_dict, *_ = reader.read()
    text = reader.extract_text(t_dict)

    res_dict = {}
    res_dict["seg_dir"] = os.path.dirname(transcript_dict["path"]).replace(
        "440K_full", "440K_seg"
    )

    if text.islower():
        res_dict["label"] = "LOWER"
    elif text.isupper():
        res_dict["label"] = "UPPER"
    elif text == "":
        res_dict["label"] = "EMPTY"
    else:
        res_dict["label"] = "MIXED"

    return res_dict

#%%
def not_lower(row):
    return not row["text"].islower()

def has_comma_period(row):
    return "," in row["text"] and "." in row["text"]
#%%
ds.filter(not_lower).count()

#%%
ds.filter(has_comma_period).count()

#%%
ds.filter(not_lower).filter(has_comma_period).count()

#%%
ds.filter(has_comma_period).filter(not_lower).count()

#%%
def mod_dict(row):
    del row["path"]
    return row
#%%
ds.filter(not_lower).filter(has_comma_period).map(mod_dict).count()

#%%
def not_lower_and_has_comma_period(row):
    return not_lower(row) and has_comma_period(row)
#%%
ds.filter(not_lower_and_has_comma_period).count()

#%%
ds.filter(not_lower_and_has_comma_period).map(mod_dict).show()
#%%
ds.filter(not_lower_and_has_comma_period).map(mod_dict).count()
# %%
ds = (
    ray.data.read_binary_files(
        paths="data/00000", file_extensions=["srt"], include_paths=True
    )
    .map(bytes_to_text)
    .map(check_case)
)

# %%
ds.show()

#%%
