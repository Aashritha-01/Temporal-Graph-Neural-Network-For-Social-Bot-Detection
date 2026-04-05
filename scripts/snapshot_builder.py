import json
import pandas as pd
import os

DATA_DIR = "data/raw/twibot20"
OUT_FILE = "data/raw/interactions.csv"

rows = []

for split in ["train.json", "dev.json", "test.json"]:
    path = os.path.join(DATA_DIR, split)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)   # list of users

    for user in data:
        user_id = user.get("ID", None)
        tweets = user.get("tweets", [])

        for tweet in tweets:
            timestamp = tweet.get("created_at", None)
            if timestamp is None:
                continue

            # 1️⃣ Retweet
            if "retweeted_status" in tweet:
                try:
                    target = tweet["retweeted_status"]["user"]["id_str"]
                    rows.append([user_id, target, timestamp])
                except:
                    pass

            # 2️⃣ Reply
            reply_to = tweet.get("in_reply_to_user_id_str", None)
            if reply_to:
                rows.append([user_id, reply_to, timestamp])

            # 3️⃣ Mentions
            mentions = tweet.get("entities", {}).get("user_mentions", [])
            for m in mentions:
                target = m.get("id_str", None)
                if target:
                    rows.append([user_id, target, timestamp])

df = pd.DataFrame(rows, columns=["user_id", "target_id", "timestamp"])

print("Total interactions extracted:", len(df))
df.to_csv(OUT_FILE, index=False)

print("✔ Interaction file created:", OUT_FILE)
