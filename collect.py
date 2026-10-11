import re
import pandas as pd
from googleapiclient.discovery import build

API_KEY = "AIzaSyAZ-SB9_BAFWR9qPtB1eDyhAizd2hZO228"
VIDEO_IDS = ["PNAzWA64VzY", "6i8xO_XLzI8","ermqfmFfVVo","PNAzWA64VzY","W_niHlfYWVk","Ue-1hlIagDE","13Uq1FcFb9w","kQpYP93pKeE","uiSPCpn7Yc4","bg6wwEJ-HjA","WrjynNiutjY"]  # the part after v= in the URL
MAX_PER_VIDEO = 2000

yt = build("youtube", "v3", developerKey=API_KEY)

def clean(t):
    t = re.sub(r"<[^>]+>", " ", t)             # strip html tags
    t = re.sub(r"http\S+|@\S+", " ", t)         # links and mentions
    t = re.sub(r"\s+", " ", t).strip()
    return t

def get_comments(video_id, limit):
    out, token = [], None
    while len(out) < limit:
        res = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=token,
            textFormat="plainText",
        ).execute()
        for it in res["items"]:
            s = it["snippet"]["topLevelComment"]["snippet"]
            out.append({
                "video_id": video_id,
                "text": clean(s["textDisplay"]),
                "likes": s["likeCount"],
            })
        token = res.get("nextPageToken")
        if not token:
            break
    return out

rows = []
for vid in VIDEO_IDS:
    try:
        rows += get_comments(vid, MAX_PER_VIDEO)
    except Exception as e:
        print(f"skip {vid}: {e}")  # comments disabled, etc.

df = pd.DataFrame(rows)
df = df[df["text"].str.len() > 3].drop_duplicates(subset="text")
df["label"] = ""   # fill: 0 = negative, 1 = neutral, 2 = positive
df.drop(columns=["video_id", "likes"]).to_csv("comments.csv", index=False, encoding="utf-8-sig")
print(len(df), "comments saved")