#!/usr/bin/env python3
"""
Video indexer and duplicate detecor
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
import logging
import numpy as np
import cv2
import torch
import open_clip
import faiss
import signal
import sys

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
from tqdm import tqdm


# Hardcoded encoder constants
# Those are model-specific - must be changed simultaneously
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512  # hardcoded

CONFIG = None
STATE = None

#
# Configuration
#
@dataclass(frozen=True)
class Config:
    fps: float = 3.0
    ignore_first_seconds: float = 20.0
    resize_width: int = 320
    max_duration : int = 0
    faiss_path : str = None

    # Duplicate detection
    detect_duplicates: bool = False
    knn_k: int = 30
    cosine_threshold: float = 0.90
    offset_bin_seconds: float = 0.5
    min_contiguous_seconds: float = 10.0
    jitter_frames: int = 1
    top_candidates: int = 10
    move_dup_path : str = None
    save_index_period : int = 100

    # FAISS HNSW
    hnsw_m: int = 32
    hnsw_ef_search: int = 64
    hnsw_ef_construction: int = 80


#
# State
#
@dataclass
class State:
    videos_processed: int = 0
    dirty : bool = False
    faiss_index = None


#
# Graceful termination/shutdown
#
def handle_termination(signum, frame):
    logging.warning("Received signal %s. Initiating graceful shutdown.", signum)
    save_and_exit(exit_code=130)

def save_and_exit(exit_code: int = 1):
    global STATE

    if STATE.dirty == True:
        logging.warning("Unsaved changes detected. Saving FAISS index before exit.")
        try:
            save_faiss()
        except Exception:
            logging.exception("Failed to save FAISS index during shutdown.")

    logging.warning("Exiting (code=%d).", exit_code)
    sys.exit(exit_code)


#
# Logging
#
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


#
# SQLite metadata DB
#
class MetaDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()

        # Indexed videos
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                status TEXT NOT NULL,
                scanned_at REAL NOT NULL,
                duplicate_of TEXT,
                note TEXT
            );
        """)

        # Indexed video frames, multiple frame IDs -> single video
        cur.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                ts REAL NOT NULL
            );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_frames_video_ts ON frames(video_id, ts);")
        self.conn.commit()

    def is_scanned(self, video_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM videos WHERE video_id = ? LIMIT 1;", (video_id,))
        return cur.fetchone() is not None

    def mark_video(self,
                   video_id: str,
                   path: str,
                   status: str,
                   duplicate_of: Optional[str] = None,
                   note: Optional[str] = None ) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO videos(video_id, path, status, scanned_at, duplicate_of, note)
            VALUES(?, ?, ?, ?, ?, ?);
        """, (video_id, path, status, time.time(), duplicate_of, note))
        self.conn.commit()

    def begin(self) -> None:
        self.conn.execute("BEGIN;")

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def insert_frames_return_ids(self, video_id: str, ts_list: List[float]) -> np.ndarray:
        """
        Inserts rows into frames table and returns the assigned frame_ids in insertion order.
        Uses a tight loop within a transaction (acceptable for pilot; works reliably everywhere).
        """
        cur = self.conn.cursor()
        ids: List[int] = []
        cur.execute("SELECT 1;")  # ensure cursor valid

        stmt = "INSERT INTO frames(video_id, ts) VALUES (?, ?);"
        for ts in ts_list:
            cur.execute(stmt, (video_id, float(ts)))
            ids.append(int(cur.lastrowid))
        return np.array(ids, dtype=np.int64)

    def get_frame_meta(self, frame_id: int) -> Tuple[str, float]:
        cur = self.conn.cursor()
        cur.execute("SELECT video_id, ts FROM frames WHERE frame_id = ?;", (int(frame_id),))
        row = cur.fetchone()
        if row is None:
            raise KeyError(frame_id)
        return str(row[0]), float(row[1])

    def get_frames_for_video(self, video_id: str) -> List[Tuple[int, float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT frame_id, ts FROM frames WHERE video_id = ? ORDER BY ts;", (video_id,))
        return [(int(r[0]), float(r[1])) for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


#
# FAISS index helpers
#
def create_or_load_faiss(index_path: str) -> faiss.Index:
    global CONFIG

    if os.path.exists(index_path):
        logging.info("Loading FAISS index: %s", index_path)
        index = faiss.read_index(index_path)
        tune_hnsw(index)
        return index

    logging.info("Creating new FAISS HNSW+IDMap index (dim=%d, M=%d, metric=IP).", EMBED_DIM, CONFIG.hnsw_m)
    base = faiss.IndexHNSWFlat(EMBED_DIM, CONFIG.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIDMap2(base)
    tune_hnsw(index)
    return index


def tune_hnsw(index: faiss.Index) -> None:
    global CONFIG

    """
    Apply HNSW parameters whether the index is base HNSW or wrapped in IDMap.
    """
    base = index
    # If wrapped, underlying index is index.index
    if hasattr(index, "index"):
        base = index.index

    if hasattr(base, "hnsw"):
        base.hnsw.efSearch = CONFIG.hnsw_ef_search
        base.hnsw.efConstruction = CONFIG.hnsw_ef_construction


def save_faiss() -> None:
    global CONFIG, STATE

    if STATE.index == None or CONFIG.faiss_path == None or STATE.dirty == False:
        return

    STATE.dirty = False

    # write into temp file and then rename
    tmp = CONFIG.faiss_path + ".tmp"
    faiss.write_index( STATE.index, tmp)
    os.replace(tmp, CONFIG.faiss_path)
    logging.info("Saved FAISS index: %s (ntotal=%d)", CONFIG.faiss_path, STATE.index.ntotal)

#
# Video discovery
#
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts" }

def list_videos(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    out: List[str] = []
    for root, _, files in os.walk(input_path):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def video_id_from_path(path: str) -> str:
    return os.path.basename(path)


#
# Frame extraction generator
#
def iter_sampled_frames(
    video_path: str,
    fps: float,
    ignore_first_seconds: float,
    resize_width: int,
    max_duration : int,
) -> Iterable[Tuple[float, np.ndarray]]:

    global CONFIG

    """
    Frame sampler using OpenCV seeking by timestamp.
    Yields (timestamp_seconds, frame_bgr).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    approx_dur = (frame_count / fps_cap) if fps_cap > 0 else None
    if approx_dur is not None:
        logging.debug("Opened %s (approx %.1fs, fps=%.2f)", video_path, approx_dur, fps_cap)
    else:
        logging.debug("Opened %s (duration unknown)", video_path)

    frame_interval = 1.0 / fps
    t = ignore_first_seconds

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()

        if not ok:
            break

        h, w = frame.shape[:2]
        if w != resize_width and w > 0:
            new_h = int(round(h * (resize_width / w)))
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

        yield (t, frame)
        t += frame_interval

        if max_duration != 0 and t > ignore_first_seconds + max_duration:
            break

    cap.release()


#
# OpenCLIP encoder (hardcoded model, dim=512)
#
class OpenCLIPEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Initializing OpenCLIP: %s (%s) on %s", CLIP_MODEL_NAME, CLIP_PRETRAINED, self.device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
        )
        model.eval().to(self.device)
        self.model = model
        self.preprocess = preprocess

    def encode_frames(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        import PIL.Image

        pil_images = []
        for f in frames_bgr:
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            rgb = f[:, :, ::-1]  # BGR -> RGB
            pil_images.append(PIL.Image.fromarray(rgb))

        batch = torch.stack([self.preprocess(im) for im in pil_images], dim=0).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        out = feats.detach().cpu().float().numpy().astype(np.float32)

        if out.shape[1] != EMBED_DIM:
            raise RuntimeError(f"Encoder returned dim {out.shape[1]} but expected {EMBED_DIM}.")

        return out

#
# Duplicate detection: offset voting + lightweight verification
#
@dataclass
class DuplicateDecision:
    is_duplicate: bool
    matched_video_id: Optional[str] = None
    matched_offset_seconds: Optional[float] = None
    votes: int = 0
    best_score: float = 0.0
    note: str = ""


def detect_duplicate_subclip(
    index: faiss.Index,
    db: MetaDB,
    video_id: str,
    q_ts: List[float],
    q_emb: np.ndarray,
    ) -> DuplicateDecision:
    if index.ntotal == 0:
        return DuplicateDecision(False, note="index_empty")

    logging.debug("Duplicate detection: FAISS search (ntotal=%d, k=%d)", index.ntotal, CONFIG.knn_k)
    D, I = index.search(q_emb, CONFIG.knn_k)  # I contains frame_ids from SQLite due to IDMap

    votes: Dict[Tuple[str, int], int] = {}
    best_sim: Dict[Tuple[str, int], float] = {}
    bin_size = CONFIG.offset_bin_seconds
    hits = 0

    for qi in range(I.shape[0]):
        qt = q_ts[qi]
        for nn in range(I.shape[1]):
            fid = int(I[qi, nn])
            sim = float(D[qi, nn])
            if fid < 0:
                continue
            if sim < CONFIG.cosine_threshold:
                continue

            cand_vid, cand_t = db.get_frame_meta(fid)
            if cand_vid == video_id:
                continue

            offset = cand_t - qt
            ob = int(np.floor(offset / bin_size))
            key = (cand_vid, ob)

            votes[key] = votes.get(key, 0) + 1
            best_sim[key] = max(best_sim.get(key, -1.0), sim)
            hits += 1

    logging.debug("Duplicate detection: %d neighbor hits >= threshold", hits)

    if not votes:
        return DuplicateDecision(False, note="no_votes")

    ranked = sorted(votes.items(), key=lambda kv: (kv[1], best_sim[kv[0]]), reverse=True)[: CONFIG.top_candidates]
    (cand_vid, ob), vcount = ranked[0]
    offset_seconds = ob * bin_size
    logging.debug("Top candidate: %s offset≈%.2fs votes=%d best_sim=%.3f",
                  cand_vid, offset_seconds, vcount, best_sim[(cand_vid, ob)])

    # Lightweight contiguity verification:
    cand_frames = db.get_frames_for_video(cand_vid)
    if not cand_frames:
        return DuplicateDecision(False, note="candidate_no_frames")

    cand_ids = np.array([fid for fid, _ in cand_frames], dtype=np.int64)
    cand_ts = np.array([t for _, t in cand_frames], dtype=np.float32)

    def nearest_idx(t: float) -> int:
        j = int(np.searchsorted(cand_ts, t))
        if j <= 0:
            return 0
        if j >= len(cand_ts):
            return len(cand_ts) - 1
        return j if abs(cand_ts[j] - t) < abs(cand_ts[j - 1] - t) else (j - 1)

    best_run = 0
    run = 0
    run_score_sum = 0.0
    best_avg = 0.0
    seconds_per_frame = 1.0 / CONFIG.fps

    for qi, qt in enumerate(q_ts):
        target_t = qt + offset_seconds
        cj = nearest_idx(target_t)

        matched = False
        best_local = -1.0

        for dj in range(-CONFIG.jitter_frames, CONFIG.jitter_frames + 1):
            jj = cj + dj
            if jj < 0 or jj >= len(cand_ids):
                continue
            aligned_fid = int(cand_ids[jj])

            nbrs = I[qi]
            sims = D[qi]
            for n in range(len(nbrs)):
                if int(nbrs[n]) == aligned_fid and float(sims[n]) >= CONFIG.cosine_threshold:
                    matched = True
                    best_local = max(best_local, float(sims[n]))
                    break
            if matched:
                break

        if matched:
            run += 1
            run_score_sum += best_local
            if run > best_run:
                best_run = run
                best_avg = run_score_sum / max(1, run)
        else:
            run = 0
            run_score_sum = 0.0

    best_seconds = best_run * seconds_per_frame
    logging.debug("Verification: best contiguous run=%d frames (≈%.2fs), avg_sim≈%.3f",
                  best_run, best_seconds, best_avg)

    if best_seconds >= CONFIG.min_contiguous_seconds:
        note = f"match={cand_vid} offset≈{offset_seconds:.2f}s run≈{best_seconds:.2f}s"
        return DuplicateDecision(
            True,
            matched_video_id=cand_vid,
            matched_offset_seconds=float(offset_seconds),
            votes=int(vcount),
            best_score=float(best_avg),
            note=note
        )

    return DuplicateDecision(False, note="verification_failed")


#
# Processing pipeline
#
def process_video(
    path: str,
    encoder: OpenCLIPEncoder,
    db: MetaDB
) -> None:

    global CONFIG, STATE

    vid = video_id_from_path(path)

    if db.is_scanned(vid):
        logging.info("SKIP already scanned: %s", vid)
        return

    logging.info("PROCESS %s", vid)
    logging.debug("Path: %s", path)

    try:
        # 1) Extract frames
        frames: List[np.ndarray] = []
        ts_list: List[float] = []
        for ts, frame in iter_sampled_frames(path, CONFIG.fps, CONFIG.ignore_first_seconds, CONFIG.resize_width, CONFIG.max_duration):
            frames.append(frame)
            ts_list.append(ts)

        if not frames:
            logging.warning("No frames extracted (after ignore window).")
            db.mark_video(vid, path, status="no_frames", note="no_frames_extracted")
            return

        logging.debug("Extracted %d frames at %.2f fps (ignore_first=%.1fs).",
                      len(frames), CONFIG.fps, CONFIG.ignore_first_seconds)

        # 2) Encode frames
        t0 = time.time()
        emb = encoder.encode_frames(frames)  # (N, 512), L2-normalized
        t1 = time.time()
        logging.debug("Encoded %d frames in %.2fs (dim=%d).", emb.shape[0], (t1 - t0), emb.shape[1])

        # 3) Duplicate detection
        if CONFIG.detect_duplicates and STATE.index.ntotal > 0:
            decision = detect_duplicate_subclip(STATE.index, db, vid, ts_list, emb)
            if decision.is_duplicate:
                logging.info("DUPLICATE: %s is duplicate/subclip of %s (%s). NOT adding.",
                             vid, decision.matched_video_id, decision.note)

                # Here you can add more logic what to do when a duplicate video is detected.
                # decision.matched_video_id is the video ID (file name) of which the current video is duplicate of;
                # path is the current video.
                if CONFIG.move_dup_path != None:
                    target_dir = f'{CONFIG.move_dup_path}/{decision.matched_video_id}'
                    os.makedirs(target_dir, exist_ok=True)
                    try:
                        os.rename( path, f'{target_dir}/{vid}' )
                        return
                    except Exception as e:
                        logging.exception("ERROR moving %s: to %s: %s", vid, target_dir, str(e))

                db.mark_video(vid, path, status="duplicate", duplicate_of=decision.matched_video_id, note=decision.note)
                return

            logging.info("No duplicate match for %s (reason=%s).", vid, decision.note)
        else:
            logging.debug("Duplicate detection skipped (disabled or empty index).")

        # 4) Add the video frames into db and save the DB and index.
        # We only do this if the video is not a duplicate
        # Use a transaction so frame rows do not persist if something fails before commit.
        db.begin()
        frame_ids = db.insert_frames_return_ids(vid, ts_list)  # np.int64 ids in insertion order
        logging.debug("Allocated %d frame_ids from SQLite: [%d..%d]",
                      len(frame_ids), int(frame_ids[0]), int(frame_ids[-1]))

        # Add to FAISS with explicit ids
        STATE.index.add_with_ids(emb, frame_ids)

        STATE.dirty = True
        STATE.videos_processed += 1

        if CONFIG.save_index_period != 0 and (STATE.videos_processed % CONFIG.save_index_period) == 0:
            save_faiss()

        db.commit()

        logging.debug("ADDED %s: %d vectors (FAISS ntotal=%d).", vid, emb.shape[0], STATE.index.ntotal)
        db.mark_video(vid, path, status="added", note=f"vectors={emb.shape[0]}")

    except Exception as e:
        # Ensure we rollback any pending DB transaction
        try:
            db.rollback()
        except Exception:
            pass
        logging.exception("ERROR processing %s: %s", vid, str(e))
        db.mark_video(vid, path, status="error", note=str(e))


#
# Main
#
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory containing videos OR a single video file path")
    p.add_argument("--db", required=True, help="SQLite DB path (metadata)")
    p.add_argument("--faiss", required=True, help="FAISS index path")
    p.add_argument("--detect-duplicates", action="store_true", help="Detect duplicates/subclips and skip adding them")
    p.add_argument("--fps", type=float, default=3.0, help="Sampling FPS (default: 3)")
    p.add_argument("--ignore-first", type=float, default=20.0, help="Ignore first N seconds (default: 20)")
    p.add_argument("--k", type=int, default=30, help="kNN for duplicate detection (default: 30)")
    p.add_argument("--cos", type=float, default=0.90, help="Cosine threshold (default: 0.90)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    p.add_argument("--max-duration", type=int, default=0, help="Maximum number of seconds to base the fingerprint on (default: full video)")
    p.add_argument("--move-duplicates-path", type=str, default=None, help="Move the detected duplicates into this path under the original subfolder")
    p.add_argument("--save-index-period", type=int, default=100, help="Save the FAISS index after each processed N videos (default: 100); 0 - only on exit")
    args = p.parse_args()

    setup_logging(args.verbose)

    global CONFIG, STATE

    CONFIG = Config(
        faiss_path=args.faiss,
        fps=args.fps,
        max_duration = args.max_duration,
        ignore_first_seconds=args.ignore_first,
        detect_duplicates=args.detect_duplicates,
        knn_k=args.k,
        cosine_threshold=args.cos,
        move_dup_path = args.move_duplicates_path,
        save_index_period = args.save_index_period
    )

    STATE = State()

    logging.info("Encoder fixed: %s/%s dim=%d", CLIP_MODEL_NAME, CLIP_PRETRAINED, EMBED_DIM)

    videos = list_videos(args.input)
    if not videos:
        logging.error("No video files found under: %s", args.input)
        return 2
    logging.info("Found %d video(s).", len(videos))

    db = MetaDB(args.db)
    STATE.index = create_or_load_faiss( args.faiss )
    encoder = OpenCLIPEncoder()

    signal.signal( signal.SIGINT, handle_termination )
    signal.signal( signal.SIGTERM, handle_termination )

    try:
        for vp in tqdm(videos, desc="Indexing"):
            process_video(vp, encoder, db)

        logging.info("Done.")
        return 0
    except Exception:
        logging.exception("Fatal error encountered.")
        save_and_exit(exit_code=1)
    finally:
        db.close()

if __name__ == "__main__":
    raise SystemExit(main())
