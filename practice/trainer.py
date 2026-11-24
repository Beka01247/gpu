#!/usr/bin/env python3
"""CUDA midterm trainer: 10 practice midterms + auto checker.

  python3 trainer.py list                     tasks and your best scores
  python3 trainer.py show 3                   print task 3
  python3 trainer.py exam                     random task, starts a 90-min timer
  python3 trainer.py check 3 serikov.cu       compile, run, compare, grade (/20)
  python3 trainer.py check 3 serikov.cu --static   rubric only, no GPU needed
  python3 trainer.py export                   write all tasks to TASKS.md
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

EXAM_MINUTES = 90
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".trainer_state.json")

COMMON = """
COMMON REQUIREMENTS (graded like the real midterm):
  - number of blocks = warp size * number of SMs (query both at runtime);
    threads per block = 256 (or another multiple of 32)
  - grid-stride loop in every kernel
  - input data filled ON THE GPU by a kernel, arrays on independent streams
  - async data migration: cudaMallocManaged + cudaMemPrefetchAsync
    (or cudaMalloc + cudaMemcpyAsync)
  - every sum is computed on the GPU with a shared-memory reduction kernel
  - __device__ function(s) for the per-element operation
  - host version of the computation + host verify function;
    print "Verify: PASSED" (or "Verify: FAILED")
  - print the hardware name; time the GPU kernel (cudaEvent) and the host
    version (std::chrono) and print both
  - free all memory, destroy streams/events; one file lastname.cu
  - use long long for sums
  - checked lines must look exactly like "Label = value"; if you print a
    label several times (e.g. small N first), the LAST occurrence is checked
"""

TASKS = {
    1: {
        "title": "Vector add / sub / mul + reduction (format of 2026 midterm-1)",
        "text": """
N = 2 << 20. Two int arrays a and b.
Fill:   a[i] = (i % 11) - 5
        b[i] = ((i * 7 + 3) % 11) - 5
Write three separate __device__ functions (add, subtract, multiply) and
compute c = a op b element-wise for each operation. Sum every c with the
reduction kernel. Implement the same on the host and verify.
Measure GPU vs CPU time of the multiplication.
Also print A, B and the three C arrays for N = 10 (like the real exam).

Checked output:
  Sum (addition) = ...
  Sum (subtraction) = ...
  Sum (multiplication) = ...
""",
        "expected": {"Sum (addition)": -6, "Sum (subtraction)": -12, "Sum (multiplication)": -4194310},
        "extra": [("3+ __device__ functions (add/sub/mul)", lambda s: len(re.findall(r"__device__", s)) >= 3)],
    },
    2: {
        "title": "Matrix multiplication (format of 2025 midterm-1)",
        "text": """
N = 1 << 8. Two N x N int matrices, row-major, flat index i in [0, N*N).
Fill:   A[i] = (i * 3) % 10 - 4
        B[i] = (i * 5 + 1) % 10 - 5
C = A x B. One thread per element of C (grid-stride over N*N),
a __device__ function computes the dot product of row r of A and
column c of B. Sum all elements of C with the reduction kernel.
Host matmul + verify; time both. Print A, B, C for N = 4 first.
Bonus (not checked): find the largest N your GPU handles.

Checked output:
  Sum (C) = ...
  C[0][0] = ...
  C[N-1][N-1] = ...
""",
        "expected": {"Sum (C)": -12582144, "C[0][0]": -504, "C[N-1][N-1]": 126},
        "extra": [],
    },
    3: {
        "title": "Integer SAXPY in place",
        "text": """
N = 1 << 22, k = 3.
Fill:   x[i] = (i % 13) - 6
        y[i] = (i % 7) - 3
Compute y = k * x + y IN PLACE with a __device__ function saxpyOp.
Sum y with the reduction kernel.
Trap: the host check needs the ORIGINAL y - keep a copy before the kernel.

Checked output:
  Sum (y) = ...
  y[12345] = ...
""",
        "expected": {"Sum (y)": -50, "y[12345]": 7},
        "extra": [],
    },
    4: {
        "title": "Dot product + max/min reduction",
        "text": """
N = 1 << 21.
Fill:   a[i] = (i * 13 % 17) - 8
        b[i] = (i * 5 % 9) - 4
p[i] = a[i] * b[i] (device function).
Dot = sum of p (reduction). Max and Min of p with a max/min version of
the shared-memory reduction; combine block results with atomicMax /
atomicMin (or a second kernel pass).

Checked output:
  Dot = ...
  Max (a*b) = ...
  Min (a*b) = ...
""",
        "expected": {"Dot": -32, "Max (a*b)": 32, "Min (a*b)": -32},
        "extra": [("max/min reduction (atomicMax/atomicMin or max()/min())",
                   lambda s: bool(re.search(r"atomicMax|atomicMin|\bmax\s*\(|\bmin\s*\(", s)))],
    },
    5: {
        "title": "C = A + B transposed, sum and trace",
        "text": """
N = 1024. N x N int matrices, row-major.
Fill:   A[r][c] = (r * 3 + c) % 10 - 5
        B[r][c] = (r + c * 2) % 9 - 4
C[r][c] = A[r][c] + B[c][r]   (grid-stride over N*N, r = i / N, c = i % N)
Sum all of C with the reduction kernel. Trace (C) = sum of the diagonal.

Checked output:
  Sum (C) = ...
  Trace (C) = ...
""",
        "expected": {"Sum (C)": -524292, "Trace (C)": -2053},
        "extra": [],
    },
    6: {
        "title": "Polynomial evaluation + counting with atomics",
        "text": """
N = 1 << 20.
Fill:   x[i] = (i % 21) - 10
y[i] = 2x^3 - 3x^2 + x - 7, computed by a __device__ int poly(int x).
Sum (y) with the reduction kernel. Count of y > 0 with atomicAdd.

Checked output:
  Sum (y) = ...
  Count (y > 0) = ...
""",
        "expected": {"Sum (y)": -122689036, "Count (y > 0)": 399456},
        "extra": [("atomicAdd for counting", lambda s: "atomicAdd" in s)],
    },
    7: {
        "title": "Histogram with shared-memory privatization",
        "text": """
N = 1 << 22.
Fill:   a[i] = ((i * 31 + 7) % 11) - 5      (values -5..5, 11 bins)
Each block builds its own histogram in __shared__ memory (atomicAdd on
shared), then adds it to the global histogram (atomicAdd on global).
Also Sum of a with the reduction kernel. Verify bins on the host.

Checked output (one line per bin, -5..5):
  Bin -5 = ...
  ...
  Bin 5 = ...
  Sum = ...
""",
        "expected": {"Bin -5": 381300, "Bin -4": 381301, "Bin -3": 381300, "Bin -2": 381301,
                     "Bin -1": 381300, "Bin 0": 381301, "Bin 1": 381300, "Bin 2": 381301,
                     "Bin 3": 381300, "Bin 4": 381300, "Bin 5": 381300, "Sum": -4},
        "extra": [("atomicAdd", lambda s: "atomicAdd" in s)],
    },
    8: {
        "title": "Sum / Min / Max of pseudo-random data",
        "text": """
N = 1 << 22. Use unsigned int arithmetic exactly like this:
        unsigned int h = ((unsigned int)i * 1103515245u + 12345u) >> 8;
        a[i] = (int)(h % 1001) - 500;
Compute Sum, Min and Max on the GPU with shared-memory reductions
(one kernel doing all three, or three kernels).

Checked output:
  Sum = ...
  Min = ...
  Max = ...
""",
        "expected": {"Sum": -30759, "Min": -500, "Max": 500},
        "extra": [],
    },
    9: {
        "title": "1D stencil",
        "text": """
N = 1 << 21.
Fill:   a[i] = (i % 9) - 4
c[i] = a[i-1] + 2 * a[i] + a[i+1], out-of-range neighbours count as 0.
Use a __device__ function for the stencil. Sum c with the reduction kernel.
Bonus (not checked): load a tile + halo into __shared__ memory.

Checked output:
  Sum (c) = ...
  c[0] = ...
  c[N-1] = ...
  c[1000] = ...
""",
        "expected": {"Sum (c)": -15, "c[0]": -11, "c[N-1]": 8, "c[1000]": -12},
        "extra": [],
    },
    10: {
        "title": "Matrix-vector multiplication",
        "text": """
N = 2048. A is N x N row-major, x has N elements.
Fill:   A[r][c] = (r * 7 + c * 3) % 11 - 5     (fill A and x on different streams)
        x[c] = (c % 5) - 2
y = A * x, one thread per row (grid-stride over rows), dot product in a
__device__ function. Sum (y) with the reduction kernel, Max (y) with a
max reduction.

Checked output:
  Sum (y) = ...
  Max (y) = ...
  y[7] = ...
""",
        "expected": {"Sum (y)": 9, "Max (y)": 14, "y[7]": 11},
        "extra": [],
    },
}


def strip_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def rubric(src):
    s = strip_comments(src)
    n_malloc = len(re.findall(r"\bcudaMalloc(Managed|Host)?\s*\(", s))
    n_free = len(re.findall(r"\bcudaFree(Host)?\s*\(", s))
    streams_created = "cudaStreamCreate" in s
    items = [
        ("grid-stride loop", 1, bool(re.search(r"blockDim\.x\s*\*\s*gridDim\.x|gridDim\.x\s*\*\s*blockDim\.x", s))),
        ("blocks = warp size * SMs", 1, bool(re.search(r"MultiProcessorCount|multiProcessorCount", s))
         and bool(re.search(r"warpSize|WarpSize|\b32\b", s))),
        ("streams (create + launch on stream)", 1, streams_created
         and bool(re.search(r"<<<[^<>;]*,[^<>;]*,[^<>;]*,[^<>;]*>>>", s))),
        ("async data migration", 1, bool(re.search(r"cudaMemPrefetchAsync|cudaMemcpyAsync", s))),
        ("clear memory", 1, n_malloc > 0 and n_free >= n_malloc
         and (not streams_created or "cudaStreamDestroy" in s)),
        ("__device__ function", 1, bool(re.search(r"__device__\s+(?!__)[\w:<>]+\s+\w+\s*\(", s))),
        ("timing GPU (cudaEvent) + CPU (chrono/clock)", 1, "cudaEventElapsedTime" in s
         and bool(re.search(r"chrono|clock\s*\(", s))),
        ("shared-memory reduction", 3, "__shared__" in s and "__syncthreads" in s
         and bool(re.search(r">>=\s*1|/=\s*2|/\s*2", s))),
    ]
    info = [
        ("hardware name printed", bool(re.search(r"cudaGetDeviceProperties|\.name", s))),
        ("error checking (cudaGetLastError / CHECK)", bool(re.search(r"cudaGetLastError|cudaError_t", s))),
        ("host verify function", bool(re.search(r"\bverify\w*\s*\(|\bcheck\w*\s*\(", s, re.I))),
    ]
    return items, info


def parse_value(out, label):
    pat = r"^\s*" + re.escape(label) + r"\s*[=:]\s*(-?\d+)"
    found = re.findall(pat, out, flags=re.M)
    return int(found[-1]) if found else None


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(st, f)
    except OSError:
        pass


def task_text(n):
    t = TASKS[n]
    return f"=== TASK {n:02d}: {t['title']} ===\n{t['text']}{COMMON}"


def cmd_list():
    st = load_state().get("best", {})
    for n, t in TASKS.items():
        best = st.get(str(n))
        mark = f"best {best:4.1f}/20" if best is not None else "not tried"
        print(f"  {n:2d}. {t['title']:<62} {mark}")


def cmd_exam():
    st = load_state()
    best = st.get("best", {})
    pool = [n for n in TASKS if best.get(str(n), 0) < 18] or list(TASKS)
    n = random.choice(pool)
    st["exam"] = {"task": n, "start": time.time()}
    save_state(st)
    print(task_text(n))
    print(f">>> Exam started: task {n}, {EXAM_MINUTES} minutes. "
          f"Check with: python3 trainer.py check {n} lastname.cu")


def cmd_check(n, path, static_only):
    if n not in TASKS:
        sys.exit(f"No task {n}")
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    if not path.endswith(".cu"):
        print("WARNING: submission must be a .cu file (-5% on the real exam)")
    src = open(path).read()
    t = TASKS[n]
    score = 0.0
    print(f"=== Checking task {n:02d}: {t['title']} ===\n")

    correct_pts = 0.0
    if static_only:
        print("[static mode] compile/run skipped, correctness not graded\n")
    else:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("nvcc not found -> run with --static, or use a machine with CUDA\n")
        else:
            tmp = tempfile.mkdtemp()
            exe = os.path.join(tmp, "prog")
            flags = os.environ.get("NVCC_FLAGS", "-O2 -std=c++17").split()
            print("$ nvcc " + " ".join(flags) + f" -o prog {path}")
            cp = subprocess.run([nvcc, *flags, "-o", exe, path], capture_output=True, text=True)
            if cp.returncode != 0:
                print(cp.stdout + cp.stderr)
                print("COMPILATION FAILED (-10% on the real exam)\n")
            else:
                if cp.stderr.strip():
                    print(cp.stderr.strip()[:1500] + "\n")
                print("$ ./prog")
                try:
                    t0 = time.time()
                    rp = subprocess.run([exe], capture_output=True, text=True, timeout=120)
                    out = rp.stdout + rp.stderr
                    lines = out.splitlines()
                    shown = lines if len(lines) <= 50 else lines[:25] + ["   ..."] + lines[-20:]
                    print("\n".join("   | " + l[:160] for l in shown))
                    print(f"   (exit code {rp.returncode}, {time.time() - t0:.1f}s)\n")
                    ok = 0
                    for label, exp in t["expected"].items():
                        got = parse_value(out, label)
                        good = got == exp
                        ok += good
                        print(f"  {'OK ' if good else 'BAD'} {label} = {got if got is not None else '(missing)'}"
                              + ("" if good else f"   expected {exp}"))
                    passed = re.search(r"PASSED", out) and not re.search(r"FAILED", out)
                    print(f"  {'OK ' if passed else 'BAD'} Verify line says PASSED")
                    frac = ok / len(t["expected"])
                    if not passed:
                        frac *= 0.8
                    if rp.returncode != 0:
                        frac *= 0.5
                    correct_pts = round(10 * frac, 1)
                except subprocess.TimeoutExpired:
                    print("RUN TIMED OUT (120s) - infinite loop or deadlock?")
            shutil.rmtree(tmp, ignore_errors=True)
    score += correct_pts
    print(f"\n  correctness: {correct_pts}/10")

    items, info = rubric(src)
    print("\nRubric:")
    for name, pts, ok in items:
        score += pts if ok else 0
        print(f"  [{'x' if ok else ' '}] {name} ({pts}%)")
    for name, fn in t["extra"]:
        ok = fn(strip_comments(src))
        print(f"  [{'x' if ok else ' '}] task-specific: {name}" + ("" if ok else "   (-1)"))
        if not ok:
            score -= 1
    for name, ok in info:
        print(f"  [{'x' if ok else ' '}] {name} (not scored)")

    score = max(score, 0)
    print(f"\nSCORE: {score:.1f}/20" + ("  (static: max 10 without correctness)" if static_only else ""))

    st = load_state()
    ex = st.get("exam")
    if ex and ex.get("task") == n:
        mins = (time.time() - ex["start"]) / 60
        late = mins - EXAM_MINUTES
        print(f"Exam time: {mins:.0f} min" + (f"  -> {late:.0f} min LATE" if late > 0 else " (in time)"))
    if not static_only:
        best = st.setdefault("best", {})
        best[str(n)] = max(best.get(str(n), 0), score)
        save_state(st)


def cmd_export():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TASKS.md")
    with open(path, "w") as f:
        f.write("# CUDA midterm practice tasks\n\n")
        for n in TASKS:
            f.write("```\n" + task_text(n).rstrip() + "\n```\n\n")
    print(f"Wrote {path}")


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help", "help"):
        print(__doc__)
    elif a[0] == "list":
        cmd_list()
    elif a[0] == "show" and len(a) > 1:
        print(task_text(int(a[1])))
    elif a[0] == "exam":
        cmd_exam()
    elif a[0] == "check" and len(a) > 2:
        cmd_check(int(a[1]), a[2], "--static" in a)
    elif a[0] == "export":
        cmd_export()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
