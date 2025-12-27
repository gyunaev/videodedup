# Video Deduplicator based on OpenCLIP and FAISS

A highly configurable video deduplicator script based on FAISS and OpenCLIP. This was built as a research project but showed a very good, reliable detection of duplicate videos, including subclip/excerpt detection. The deduplication process is also resilient to repacking, format conversion, resize and subtitle/logo embedding. It could use GPU (cuda) if available, but otherwise shall fallback on CPU.

## Installation
To install:

- clone the repository;
- set up Python virtual environment;
- install dependencies: `pip install -r requirements.txt`;
- run the script: `python3 videodup.py --input MyVideos --db videos.db --faiss faiss.idx --detect-duplicates`;

## Usage
The following parameters are mandatory:
- `--input` specifies a directory. All videos in this directory will be indexed and added into the database.
- `--db` specifies a full pathname where to store the database of scanned videos
- `--faiss` specifies a full pathname where to store the FAISS index
- `--detect-duplicates` by default the script only indexes the videos without detecting duplicates. If this option is chosen, before adding each video it tries to detect if it is a duplicate, in which case it does not add it. The detected duplicates are printed on the screen.
- `--ignore-first <N>` tells the script to ignore the first N seconds of the video (useful to detect duplicates when for example an intro logo was cut).
- `--max-duration <M>` by detaul the script indexes the whole video (which may lead to OOM as stated below). Indexing the whole video is only necessary in specific scenarios, and slows down the processing significantly. You should use this option to index - and detect duplicates - only in the first M seconds of the video (after N, i.e. specifying `--ignore-first 10` and `--max-duration 20` would mean the seconds 10-30 of the video are indexed).
- `--save-index-period <S>` save the index after each S new videos added. Saving the index takes a while (you can process 10+ videos while a large index is saved), thus on large pipelines it should be set to a high enough value, like 100.

## Performance
The script performance directly depends on how many frames are indexed (`--max-duration` option). With 20 seconds it can process 2-5 videos per second. Saving the index slows the processing down.

## Limitations
- This is a research project; treat accordingly.
- The script uses the video file name as a unique identifier. In a single directory this would not collide, but if this is not the case for you, change the `video_id_from_path` function accordingly (i.e. use either absolute path, or - to support multi-system database - a machine/path/filename or a `sha256` hash of the content).
- If you do not use `--max-duration` on larger videos you'll likely run out of VRAM/RAM. If you need to fully analyze large videos, reimplement `encoded_frames` to support frame batching.

## Internals
The script works as following:

1. Using OpenCV (`cv2.VideoCapture`) to capture the specific number of frames from a video file.
2. Using [OpenCLIP](https://github.com/mlfoundations/open_clip) to encode each frame into a 512-dim vector, associating each frame with unique frame_id (associated with video);
3. Using [FAISS](https://github.com/facebookresearch/faiss) to store the vectors and find similarity between vectors.
4. FAISS configuration uses HNSW (simpler and requires no training); Duplicate verification is lightweight (neighbor-membership + contiguity for `min_contiguous_seconds`). For stricter verification, we should store embeddings and compute exact cosine over aligned frames. This however would increase the index size dramatically.
5. FAISS runs on CPU due to the index growing fast (roughly 1Gb per 5k processed videos). If you have enough VRAM available, replacing `faiss_cpu` by `faiss_gpu` (and installing relevant dependencies) would speed the matching up.
