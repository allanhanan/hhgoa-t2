import json
import pyarrow.parquet as pq
import fsspec

URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/tamtrain.parquet"

fs = fsspec.filesystem("https")
pf = pq.ParquetFile(fs.open(URL, "rb"))

print("Reading 10 development rows...")

batch = next(
    pf.iter_batches(
        batch_size=10,
        columns=[
            "query",
            "Eng_Query",
            "Answer",
            "query_id",
            "passages.English_passages",
        ],
        use_threads=False,
    )
)

rows = batch.to_pylist()

with open(
    "data/dev/msmarco_xi_tamil_sample.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Saved {len(rows)} rows")
