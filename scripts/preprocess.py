import json
import pandas as pd
import os

DATA_DIR = "data/raw/twibot20"
OUT_FILE = "data/raw/interactions.csv"

rows = []

for split in ["train.json", "dev.json", "test.json"]:
    path = os.path.join(DATA_DIR, split)
    print("Reading:", path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for user in data:
        user_id = user.get("ID", None)
        tweets = user.get("tweets", [])

        for tweet in tweets:
            ts = tweet.get("created_at")
            if not ts:
                continue

            # replies
            reply_to = tweet.get("in_reply_to_user_id_str")
            if reply_to:
                rows.append([user_id, reply_to, ts])

            # mentions
            mentions = tweet.get("entities", {}).get("user_mentions", [])
            for m in mentions:
                if "id_str" in m:
                    rows.append([user_id, m["id_str"], ts])

            # retweets
            if "retweeted_status" in tweet:
                try:
                    rows.append([
                        user_id,
                        tweet["retweeted_status"]["user"]["id_str"],
                        ts
                    ])
                except:
                    pass

df = pd.DataFrame(rows, columns=["user_id", "target_id", "timestamp"])

print("Total interactions extracted:", len(df))
df.to_csv(OUT_FILE, index=False)
print("✔ Interaction file created:", OUT_FILE)
