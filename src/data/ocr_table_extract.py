"""OCR Table Extraction Module for Scanned Parliamentary PDF Documents.

Provides OCR table and prose extraction using Tesseract with:
- Zero-disk, pure in-memory streaming IPC (32 KB chunked Pipe).
- Cross-platform process-tree hard timeout (Linux/macOS process group SIGKILL, Windows taskkill /F /T).
- Spatial line clustering and robust column-gutter detection.
- Reliable Table-vs-Prose structure classifier.
- Strict Dependency Policy: raises DependencyMissingError when Tesseract/pytesseract is missing.
"""

from __future__ import annotations

import io
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from typing import Any

# Automatically register local fallback paths if present
_LOCAL_PATHS = [
    ("/home/user/opt/tesseract/usr/bin", "/home/user/opt/tesseract/usr/lib/x86_64-linux-gnu", "/home/user/opt/tesseract/usr/share/tesseract-ocr/5/tessdata"),
    ("/home/user/.local/usr/bin", "/home/user/.local/usr/lib/x86_64-linux-gnu", "/home/user/.local/usr/share/tesseract-ocr/5/tessdata"),
]

for _bin_dir, _lib_dir, _tess_dir in _LOCAL_PATHS:
    if os.path.isdir(_bin_dir) and _bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_bin_dir}:{os.environ.get('PATH', '')}"
    if os.path.isdir(_lib_dir) and _lib_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{_lib_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if os.path.isdir(_tess_dir) and "TESSDATA_PREFIX" not in os.environ:
        os.environ["TESSDATA_PREFIX"] = _tess_dir


class DependencyMissingError(ImportError, RuntimeError):
    """Raised when a mandatory dependency (PyMuPDF or Tesseract OCR) is unavailable."""
    pass


class ExtractionTimeoutError(RuntimeError):
    """Raised when an OCR extraction operation exceeds its hard time ceiling and is killed."""
    pass


try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    for _bin_dir, _, _ in _LOCAL_PATHS:
        if os.path.exists(f"{_bin_dir}/tesseract_runner"):
            pytesseract.pytesseract.tesseract_cmd = f"{_bin_dir}/tesseract_runner"
            break
        elif os.path.exists(f"{_bin_dir}/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = f"{_bin_dir}/tesseract"
            break
    _OCR_LIBS_AVAILABLE = True
except ImportError:
    _OCR_LIBS_AVAILABLE = False


def is_tesseract_available() -> bool:
    """Check if pytesseract and Tesseract OCR engine binary are operational."""
    if not _OCR_LIBS_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Grayscale conversion and mild contrast normalization."""
    gray = img.convert("L")
    enhancer = ImageEnhance.Contrast(gray)
    return enhancer.enhance(1.2)


_IPC_CHUNK_SIZE = 32768  # 32 KB chunk size for continuous in-memory pipe draining


def _kill_process_tree(pid: int, proc: multiprocessing.Process | None = None) -> None:
    """Cross-platform process-tree termination for Linux, macOS, and Windows."""
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
        if proc and proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            if proc:
                try:
                    proc.terminate()
                except OSError:
                    pass
        time.sleep(0.05)
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            if proc:
                try:
                    proc.kill()
                except OSError:
                    pass


def _ocr_process_entrypoint(img_bytes: bytes, dpi: int, timeout: float, conn):
    """Subprocess target executing Tesseract OCR inside an isolated process group streaming in-memory chunks."""
    try:
        if sys.platform != "win32":
            try:
                os.setpgrp()
            except Exception:
                pass
            
        image = Image.open(io.BytesIO(img_bytes))
        result = ocr_image_to_structured_text(image, timeout=timeout)
        
        # Stream structured result in memory-bounded chunks without filesystem I/O
        for i in range(0, len(result), _IPC_CHUNK_SIZE):
            conn.send(("CHUNK", result[i:i + _IPC_CHUNK_SIZE]))
        conn.send(("SUCCESS", ""))
    except Exception as exc:
        conn.send(("ERROR", str(exc)))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ocr_page_to_structured_text(page, dpi: int = 300, timeout: float = 10.0) -> str:
    """Rasterize page and run OCR in an isolated subprocess with continuously drained in-memory IPC and cross-platform termination."""
    if not is_tesseract_available():
        raise DependencyMissingError("Tesseract OCR is not installed or pytesseract is unavailable.")

    pix = page.get_pixmap(dpi=dpi)
    img_bytes = pix.tobytes("png")

    parent_conn, child_conn = multiprocessing.Pipe()
    proc = multiprocessing.Process(
        target=_ocr_process_entrypoint,
        args=(img_bytes, dpi, timeout, child_conn)
    )
    proc.start()
    pid = proc.pid

    # Close parent copy of child pipe endpoint so worker EOF is clean
    try:
        child_conn.close()
    except Exception:
        pass

    chunks: list[str] = []
    deadline = time.time() + timeout
    result_str: str | None = None

    try:
        while time.time() < deadline:
            remaining = max(0.001, deadline - time.time())
            try:
                has_data = parent_conn.poll(min(0.02, remaining))
            except (EOFError, BrokenPipeError, OSError):
                # Worker closed pipe on completion/exit
                break

            if has_data:
                try:
                    msg_type, payload = parent_conn.recv()
                except (EOFError, BrokenPipeError, OSError):
                    break

                if msg_type == "CHUNK":
                    chunks.append(payload)
                elif msg_type == "SUCCESS":
                    result_str = "".join(chunks)
                    break
                elif msg_type == "ERROR":
                    if "DependencyMissingError" in payload:
                        raise DependencyMissingError(payload)
                    raise RuntimeError(f"Subprocess OCR failed: {payload}")
            elif not proc.is_alive():
                # Process exited; drain any remaining buffered chunks
                try:
                    while parent_conn.poll():
                        msg_type, payload = parent_conn.recv()
                        if msg_type == "CHUNK":
                            chunks.append(payload)
                        elif msg_type == "SUCCESS":
                            result_str = "".join(chunks)
                            break
                        elif msg_type == "ERROR":
                            if "DependencyMissingError" in payload:
                                raise DependencyMissingError(payload)
                            raise RuntimeError(f"Subprocess OCR failed: {payload}")
                except (EOFError, BrokenPipeError, OSError):
                    pass
                break

        # If SUCCESS was received, allow worker up to 2.0s to finish clean exit
        if result_str is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                _kill_process_tree(pid, proc)
                proc.join(timeout=0.5)
            return result_str

        # If not completed and process still alive after deadline, terminate
        proc.join(timeout=0.1)
        if proc.is_alive():
            _kill_process_tree(pid, proc)
            proc.join(timeout=0.5)
            raise TimeoutError(f"Hard OCR timeout exceeded (>{timeout}s); worker process tree PID {pid} was terminated.")

        if result_str is not None:
            return result_str
        return "".join(chunks)
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass


def ocr_image_to_structured_text(image: Image.Image, timeout: float | None = None) -> str:
    """Extract structured Markdown tables and text from a PIL Image with spatial clustering and table-vs-prose classification."""
    if not is_tesseract_available():
        raise DependencyMissingError("Tesseract OCR is not installed or pytesseract is unavailable.")
    
    processed_img = preprocess_image_for_ocr(image)
    
    ocr_kwargs: dict[str, Any] = {
        "output_type": pytesseract.Output.DICT,
        "config": "--oem 1 --psm 6"
    }
    if timeout is not None and timeout > 0:
        ocr_kwargs["timeout"] = timeout

    ocr_data = pytesseract.image_to_data(processed_img, **ocr_kwargs)
    
    words = []
    n_boxes = len(ocr_data["text"])
    for i in range(n_boxes):
        text = str(ocr_data["text"][i]).strip()
        conf = float(ocr_data["conf"][i])
        if text and conf > 20:
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]
            words.append({
                "text": text,
                "x0": x,
                "x1": x + w,
                "y0": y,
                "y1": y + h,
                "yc": y + h / 2.0,
                "conf": conf
            })
    if not words:
        return ""

    # Sort words by vertical center
    words.sort(key=lambda w: w["yc"])
    lines: list[list[dict[str, Any]]] = []
    current_line = [words[0]]
    for w in words[1:]:
        line_yc = sum(item["yc"] for item in current_line) / len(current_line)
        line_height = sum(item["y1"] - item["y0"] for item in current_line) / len(current_line)
        if abs(w["yc"] - line_yc) <= max(line_height * 0.5, 8.0):
            current_line.append(w)
        else:
            current_line.sort(key=lambda item: item["x0"])
            lines.append(current_line)
            current_line = [w]
    if current_line:
        current_line.sort(key=lambda item: item["x0"])
        lines.append(current_line)

    structured_blocks = []
    table_candidate_lines = []

    for line in lines:
        is_table_line = False
        if len(line) >= 2:
            gaps = [line[i]["x0"] - line[i-1]["x1"] for i in range(1, len(line))]
            if any(gap > 45.0 for gap in gaps):
                is_table_line = True

        if is_table_line:
            table_candidate_lines.append(line)
        else:
            if len(table_candidate_lines) >= 2:
                table_md = _reconstruct_ocr_table(table_candidate_lines)
                if table_md:
                    structured_blocks.append(table_md)
                else:
                    for tl in table_candidate_lines:
                        structured_blocks.append(" ".join(w["text"] for w in tl))
                table_candidate_lines = []
            elif table_candidate_lines:
                for tl in table_candidate_lines:
                    structured_blocks.append(" ".join(w["text"] for w in tl))
                table_candidate_lines = []

            structured_blocks.append(" ".join(w["text"] for w in line))

    if len(table_candidate_lines) >= 2:
        table_md = _reconstruct_ocr_table(table_candidate_lines)
        if table_md:
            structured_blocks.append(table_md)
        else:
            for tl in table_candidate_lines:
                structured_blocks.append(" ".join(w["text"] for w in tl))
    elif table_candidate_lines:
        for tl in table_candidate_lines:
            structured_blocks.append(" ".join(w["text"] for w in tl))

    return "\n\n".join(b for b in structured_blocks if b.strip())


def _reconstruct_ocr_table(lines: list[list[dict]]) -> str | None:
    """Infer column boundaries via spatial clustering and render Markdown table if tabular."""
    all_words = [w for line in lines for w in line]
    if not all_words:
        return None

    xs = sorted(w["x0"] for w in all_words)
    col_starts = []
    for x in xs:
        if not col_starts:
            col_starts.append(x)
        elif x - col_starts[-1] > 55:
            col_starts.append(x)

    if len(col_starts) < 2:
        return None

    def find_col(x: float) -> int:
        return min(range(len(col_starts)), key=lambda i: abs(col_starts[i] - x))

    rows = []
    for line in lines:
        cols_content = {i: [] for i in range(len(col_starts))}
        for w in line:
            c_idx = find_col(w["x0"])
            cols_content[c_idx].append(w["text"])
        
        row_cells = [" ".join(cols_content[i]).strip() for i in range(len(col_starts))]
        if any(row_cells):
            rows.append(row_cells)

    if len(rows) < 2:
        return None

    # Structural Table-vs-Prose Verification: Ensure at least 50% of rows have >= 2 filled columns
    multi_col_rows = sum(1 for r in rows if sum(1 for c in r if c) >= 2)
    if multi_col_rows / len(rows) < 0.5:
        return None

    hdr = rows[0]
    sep = ["---"] * len(hdr)
    md_lines = [
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(sep) + " |"
    ]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")

    return "\n".join(md_lines)
