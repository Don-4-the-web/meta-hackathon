import asyncio
import json
import sqlite3

from app.config import settings

settings.meta_api_key = ""
from app import pipeline  # noqa: E402

db = sqlite3.connect("chat.db")
row = db.execute(
    "SELECT id, name, description, dna FROM assets WHERE name LIKE 'Norman%'"
).fetchone()
db.close()
asset = {"id": row[0], "kind": "character", "name": row[1],
         "description": row[2], "version": 2, "dna": json.loads(row[3])}
print("DNA:", row[3])
r = asyncio.run(pipeline.run_pipeline("Norman tends his garden.", 6, 42, [asset]))
svg = r["panels"][0]["svg"]
print("score:", r["panels"][0]["consistency_score"])
i = svg.find("<g transform")
print(svg[i:i + 1300])
