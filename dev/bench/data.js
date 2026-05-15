window.BENCHMARK_DATA = {
  "lastUpdate": 1778854133705,
  "repoUrl": "https://github.com/adrhill/asdex",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c71c658ab58a0565403ced62a3e1f434d3a2039e",
          "message": "Redesign benchmarks with realistic test functions (#15)\n\n* Redesign benchmarks with realistic test functions\n\nReplace synthetic benchmarks with 3 realistic test cases:\n- Heat equation RHS: tridiagonal Jacobian (~3 colors, 98% sparse)\n- Pure ConvNet: 3 conv layers with ReLU (~18 colors, 95% sparse)\n- Rosenbrock function: sparse Hessian (~3 colors, 98% sparse)\n\nEach function now has 4 benchmark phases: detection, coloring,\nmaterialization (with pre-computed sparsity/colors), and end-to-end.\n\nAlso add TODO for custom_vjp/custom_jvp support (needed for jax.nn.relu).\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add Hessian tests and improve coverage to 99%\n\nAdd 6 new tests for sparse_hessian:\n- test_hessian_quadratic: basic quadratic function\n- test_hessian_rosenbrock: sparse tridiagonal pattern\n- test_hessian_precomputed_sparsity: pre-computed sparsity\n- test_hessian_precomputed_colors: pre-computed colors\n- test_hessian_zero: linear function (zero Hessian)\n- test_hessian_single_input: single dimension\n\nAlso move hessian_sparsity import to module level in decompression.py.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-06T15:18:46+01:00",
          "tree_id": "d4fc63846af52bfae27dce0aa469a6234eaff27f",
          "url": "https://github.com/adrhill/asdex/commit/c71c658ab58a0565403ced62a3e1f434d3a2039e"
        },
        "date": 1770387596647,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 444.8577397523808,
            "unit": "iter/sec",
            "range": "stddev: 0.00019287888722053595",
            "extra": "mean: 2.247909636362909 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 1279.1086699046168,
            "unit": "iter/sec",
            "range": "stddev: 0.00001873059124719543",
            "extra": "mean: 781.7944038129067 usec\nrounds: 1154"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 25.681951608896146,
            "unit": "iter/sec",
            "range": "stddev: 0.006590553478836575",
            "extra": "mean: 38.937850799999296 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 29.61872394234056,
            "unit": "iter/sec",
            "range": "stddev: 0.0004703285107900902",
            "extra": "mean: 33.7624268333343 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.789696478696747,
            "unit": "iter/sec",
            "range": "stddev: 0.014393870357419097",
            "extra": "mean: 50.531345999999644 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 144.84349241414247,
            "unit": "iter/sec",
            "range": "stddev: 0.000059335944312373055",
            "extra": "mean: 6.904003647887466 msec\nrounds: 142"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.74565583610344,
            "unit": "iter/sec",
            "range": "stddev: 0.0015947026986007455",
            "extra": "mean: 364.21170739999695 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.3563527405611375,
            "unit": "iter/sec",
            "range": "stddev: 0.018450783950387448",
            "extra": "mean: 424.3846784000013 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 10.543084128788397,
            "unit": "iter/sec",
            "range": "stddev: 0.03464758780697518",
            "extra": "mean: 94.84890642857076 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 4.681852591391767,
            "unit": "iter/sec",
            "range": "stddev: 0.003212477475886659",
            "extra": "mean: 213.5906631999987 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.43612065135260547,
            "unit": "iter/sec",
            "range": "stddev: 0.012616653719753667",
            "extra": "mean: 2.2929434708 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.3839516882670346,
            "unit": "iter/sec",
            "range": "stddev: 0.03348593097557271",
            "extra": "mean: 2.604494342799998 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ced92bb4520950b819640b6457dbbee4b71cae07",
          "message": "Add `SparsityPattern` data structure (#16)\n\n* Add SparsityPattern data structure to replace BCOO in pipeline\n\nReplace BCOO with a custom SparsityPattern class optimized for the\ndetection->coloring->decompression pipeline:\n\n- Store row/col indices separately for direct access (no slicing)\n- Cache col_to_rows mapping for coloring algorithm\n- No unnecessary data values (pattern-only, no all-1s array)\n- Julia-style visualization: dots for small matrices, braille for large\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update README to showcase SparsityPattern visualization\n\n- Jacobian example: n=100 with braille display\n- Hessian example: n=5 diagonal pattern with dot display\n- Show SparsityPattern pretty-printing output\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Use Julia-style bracket characters for braille frame\n\nBox-drawing characters (┌─┐│└┘) have inconsistent width with braille\nin some fonts. Switch to mathematical bracket characters (⎡⎢⎣⎤⎥⎦)\nwhich align correctly, matching Julia's SparseArrays visualization.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-06T17:07:14+01:00",
          "tree_id": "73e1e1c2de7a7d140f94137e699577d78d38177e",
          "url": "https://github.com/adrhill/asdex/commit/ced92bb4520950b819640b6457dbbee4b71cae07"
        },
        "date": 1770394106346,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 658.6643183407812,
            "unit": "iter/sec",
            "range": "stddev: 0.0038036857844234674",
            "extra": "mean: 1.5182240363635697 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3386.9774510275793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000273161587884041",
            "extra": "mean: 295.2484964718643 usec\nrounds: 1984"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 33.45235659508205,
            "unit": "iter/sec",
            "range": "stddev: 0.000571204862102254",
            "extra": "mean: 29.893260200000782 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 31.76640590225183,
            "unit": "iter/sec",
            "range": "stddev: 0.00020689727169111754",
            "extra": "mean: 31.479796709677903 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.015167176784573,
            "unit": "iter/sec",
            "range": "stddev: 0.013426086562642568",
            "extra": "mean: 47.58467974999974 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.24665497798958,
            "unit": "iter/sec",
            "range": "stddev: 0.00021621249484248735",
            "extra": "mean: 4.077527581730924 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.7824130863022223,
            "unit": "iter/sec",
            "range": "stddev: 0.002020764051644127",
            "extra": "mean: 359.4002647999986 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.446697402777567,
            "unit": "iter/sec",
            "range": "stddev: 0.0026561263904926846",
            "extra": "mean: 408.71421159999954 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 44.85053583281214,
            "unit": "iter/sec",
            "range": "stddev: 0.009383748701236745",
            "extra": "mean: 22.296277657142536 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.566664731879604,
            "unit": "iter/sec",
            "range": "stddev: 0.0007132352750473701",
            "extra": "mean: 179.6407809999986 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4377555001289671,
            "unit": "iter/sec",
            "range": "stddev: 0.022375799519721178",
            "extra": "mean: 2.284380207000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.40402917248649256,
            "unit": "iter/sec",
            "range": "stddev: 0.03129882212142588",
            "extra": "mean: 2.475068802199999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e6443d223df8fe6343fe597a27ebeb1a642186df",
          "message": "Support gather, scatter, and custom JVP/VJP (#17)\n\n- Add prop_gather and prop_scatter with static index support\n- Add prop_custom_call to trace into custom_jvp_call/custom_vjp_call\n- Add const_vals tracking to propagate constant values through primitives\n- This enables precise sparsity patterns for static indexing operations\n  (e.g., x[[2,0,1]] produces permutation matrix, not all-ones)\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T01:46:33+01:00",
          "tree_id": "8983067a190da81e21583bf6658b1b368c74b0f1",
          "url": "https://github.com/adrhill/asdex/commit/e6443d223df8fe6343fe597a27ebeb1a642186df"
        },
        "date": 1770425266154,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 694.1697879670656,
            "unit": "iter/sec",
            "range": "stddev: 0.0034795646168398415",
            "extra": "mean: 1.4405697530118444 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3380.2252265282987,
            "unit": "iter/sec",
            "range": "stddev: 0.00000717572099927998",
            "extra": "mean: 295.83827496224626 usec\nrounds: 1993"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 32.66335202019078,
            "unit": "iter/sec",
            "range": "stddev: 0.0006772045378707155",
            "extra": "mean: 30.615351399998758 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 30.091325574516297,
            "unit": "iter/sec",
            "range": "stddev: 0.00938402662080791",
            "extra": "mean: 33.23216843750076 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.997422878895645,
            "unit": "iter/sec",
            "range": "stddev: 0.011445354025197571",
            "extra": "mean: 43.483133099999804 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.67453114080476,
            "unit": "iter/sec",
            "range": "stddev: 0.00002841045992376862",
            "extra": "mean: 4.18980607281493 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.4775076237232847,
            "unit": "iter/sec",
            "range": "stddev: 0.0029116580244819743",
            "extra": "mean: 403.6314521999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.2149689441441436,
            "unit": "iter/sec",
            "range": "stddev: 0.004243021060255774",
            "extra": "mean: 451.4735986000005 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 43.53190475705649,
            "unit": "iter/sec",
            "range": "stddev: 0.013760512826358403",
            "extra": "mean: 22.97165735294183 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.30469303159988,
            "unit": "iter/sec",
            "range": "stddev: 0.001979586878893617",
            "extra": "mean: 188.51232183333386 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.44794935019352644,
            "unit": "iter/sec",
            "range": "stddev: 0.012377084831477118",
            "extra": "mean: 2.2323952464000056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.40845881396684586,
            "unit": "iter/sec",
            "range": "stddev: 0.025552567374849934",
            "extra": "mean: 2.448227252800007 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "74543ac283f7bb7097cb432c38a7eb5c8eef61a1",
          "message": "Add 28 trivial primitives (#18)\n\n* Add ~28 trivial primitives to prop_dispatch\n\nExtend interpreter coverage with primitives that map directly to\nexisting handler patterns: 15 unary elementwise (inverse trig, cbrt,\nrsqrt, square, exp2, logistic, etc.), 4 binary elementwise (atan2, rem,\nnextafter, complex), 3 bitwise ops, 4 zero-derivative (argmax, argmin,\nclz, population_count), 3 identity pass-through (bitcast_convert_type,\nreduce_precision, stop_gradient), and a dedicated iota handler with\nconst-value tracking for downstream gather/scatter precision.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Increase test coverage to 99%\n\nAdd tests for edge cases and defensive branches:\n- Error paths for missing jaxpr/call_jaxpr params and unknown primitives\n- prop_jaxpr default const_vals, stop_gradient, reshape size mismatch\n- Dynamic and 2D scatter conservative fallback\n- SparsityPattern validation, empty to_bcoo, large zero-dim braille\n- Extend SymPy random expressions with log, atan, asinh\n\nThe only uncovered lines (5/680) are unreachable scatter branches\nwhere JAX validates index/updates shape consistency before our code.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T02:38:13+01:00",
          "tree_id": "5de5c6f92f46fb77be9a2219511ca4ee367370bf",
          "url": "https://github.com/adrhill/asdex/commit/74543ac283f7bb7097cb432c38a7eb5c8eef61a1"
        },
        "date": 1770428360610,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 736.2016470094069,
            "unit": "iter/sec",
            "range": "stddev: 0.0032230216490377315",
            "extra": "mean: 1.35832350289108 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3598.5782161012066,
            "unit": "iter/sec",
            "range": "stddev: 0.000014295369922670833",
            "extra": "mean: 277.88752666974847 usec\nrounds: 2081"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 35.55063673785402,
            "unit": "iter/sec",
            "range": "stddev: 0.0005334772689661583",
            "extra": "mean: 28.12889140000152 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 34.06552626613208,
            "unit": "iter/sec",
            "range": "stddev: 0.00020895368934754175",
            "extra": "mean: 29.35519011764686 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.019196270095,
            "unit": "iter/sec",
            "range": "stddev: 0.013898650916275143",
            "extra": "mean: 43.44200328571562 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 267.47898505977855,
            "unit": "iter/sec",
            "range": "stddev: 0.00009020860668547253",
            "extra": "mean: 3.7386114642857318 msec\nrounds: 224"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.549564101051154,
            "unit": "iter/sec",
            "range": "stddev: 0.0012235878756469702",
            "extra": "mean: 392.2239097999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.2788382011040076,
            "unit": "iter/sec",
            "range": "stddev: 0.0026344636648777645",
            "extra": "mean: 438.82009680000067 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 49.745322444144506,
            "unit": "iter/sec",
            "range": "stddev: 0.010637518482229845",
            "extra": "mean: 20.10239256410146 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.682975248494706,
            "unit": "iter/sec",
            "range": "stddev: 0.0001725820791701896",
            "extra": "mean: 175.96416599999762 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4969696530448073,
            "unit": "iter/sec",
            "range": "stddev: 0.014096172558825823",
            "extra": "mean: 2.012195299800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.44706400057682355,
            "unit": "iter/sec",
            "range": "stddev: 0.01239900780296027",
            "extra": "mean: 2.2368162023999956 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8e7a89254eca1fef6ff552de025165ae85d3a22f",
          "message": "Fix braille rendering and update README (#19)\n\nRewrite _render_braille to use Julia-style linear interpolation instead\nof block-OR downsampling, which made sparse patterns look artificially\ndense. Each non-zero is now scaled individually to the output grid,\nmatching Julia's SparseArrays braille display.\n\nAlso update README: use n=50 example, fix typos, add\nSparseMatrixColorings.jl to related work.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T03:12:43+01:00",
          "tree_id": "da9db562770e5d87075ef0236b6b7e7b686921c9",
          "url": "https://github.com/adrhill/asdex/commit/8e7a89254eca1fef6ff552de025165ae85d3a22f"
        },
        "date": 1770430434170,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 632.6359101279828,
            "unit": "iter/sec",
            "range": "stddev: 0.004989920799901693",
            "extra": "mean: 1.5806880134226005 msec\nrounds: 149"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3375.452495076986,
            "unit": "iter/sec",
            "range": "stddev: 0.000019369329820816824",
            "extra": "mean: 296.2565764022677 usec\nrounds: 2068"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 33.47028951847288,
            "unit": "iter/sec",
            "range": "stddev: 0.0006378333460978401",
            "extra": "mean: 29.87724380000003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 32.05348280668388,
            "unit": "iter/sec",
            "range": "stddev: 0.0003232185976418458",
            "extra": "mean: 31.19785784374973 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.4763094539022,
            "unit": "iter/sec",
            "range": "stddev: 0.012531810323954727",
            "extra": "mean: 44.49128990909075 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.06784583679834,
            "unit": "iter/sec",
            "range": "stddev: 0.00004461178809096297",
            "extra": "mean: 4.080502672986095 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.413456895420666,
            "unit": "iter/sec",
            "range": "stddev: 0.009399210465201339",
            "extra": "mean: 414.3434266000014 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.0929974236503535,
            "unit": "iter/sec",
            "range": "stddev: 0.02821716685887922",
            "extra": "mean: 477.78367460000055 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 42.51216505341042,
            "unit": "iter/sec",
            "range": "stddev: 0.014668885374465038",
            "extra": "mean: 23.522678714284343 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.307858176014629,
            "unit": "iter/sec",
            "range": "stddev: 0.005045169364548127",
            "extra": "mean: 188.39990950000166 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4430832314347581,
            "unit": "iter/sec",
            "range": "stddev: 0.021892536267014966",
            "extra": "mean: 2.256912311399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.40166510174743103,
            "unit": "iter/sec",
            "range": "stddev: 0.0462404923428099",
            "extra": "mean: 2.4896362557999994 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d8abf294b5d4ddae8c812d2e644ef0ac8d8c476a",
          "message": "Support multi-dimensional input and output shapes (#20)\n\nRename `n` parameter to `input_shape: int | tuple[int, ...]` in\n`jacobian_sparsity` and `hessian_sparsity`, allowing functions on\nmatrices and higher-dimensional arrays. Update `sparse_jacobian` and\n`sparse_hessian` to reshape VJP seeds and HVP tangents to match the\nactual input/output shapes, flattening the results for the sparse matrix.\n\nAdd `tests/test_multidim.py` with 15 tests including a tiny LeNet convnet\non 2D image input and multi-dimensional output correctness checks.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T03:59:33+01:00",
          "tree_id": "fcd47f0333d6b530b2b096b90a97b33d8c9cce83",
          "url": "https://github.com/adrhill/asdex/commit/d8abf294b5d4ddae8c812d2e644ef0ac8d8c476a"
        },
        "date": 1770433247037,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 677.0611109918782,
            "unit": "iter/sec",
            "range": "stddev: 0.004061364787030313",
            "extra": "mean: 1.476971552146931 msec\nrounds: 163"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3386.6689157342335,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070882499742896586",
            "extra": "mean: 295.27539446034064 usec\nrounds: 2094"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 32.40890601922615,
            "unit": "iter/sec",
            "range": "stddev: 0.0005419426385338814",
            "extra": "mean: 30.855716000002076 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 30.88782834943404,
            "unit": "iter/sec",
            "range": "stddev: 0.00032284454246964475",
            "extra": "mean: 32.37521229032352 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.801689971334365,
            "unit": "iter/sec",
            "range": "stddev: 0.01978838608618007",
            "extra": "mean: 48.073017210526814 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.23554974659962,
            "unit": "iter/sec",
            "range": "stddev: 0.00020534673833592633",
            "extra": "mean: 4.128213224880043 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.393174173306563,
            "unit": "iter/sec",
            "range": "stddev: 0.007365247221694762",
            "extra": "mean: 417.855086000003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.1573432906933694,
            "unit": "iter/sec",
            "range": "stddev: 0.002967856038761839",
            "extra": "mean: 463.53308920000416 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 42.20352927158153,
            "unit": "iter/sec",
            "range": "stddev: 0.015042547078038955",
            "extra": "mean: 23.69470082857187 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.072152789405264,
            "unit": "iter/sec",
            "range": "stddev: 0.0010596527878468382",
            "extra": "mean: 197.1549441666672 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4389887497409245,
            "unit": "iter/sec",
            "range": "stddev: 0.010187656164640688",
            "extra": "mean: 2.2779627054000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.4007114364841244,
            "unit": "iter/sec",
            "range": "stddev: 0.008588270173646582",
            "extra": "mean: 2.4955614163999997 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b9c2deaedc549dbf6297e04dac3dc07ea9c224e0",
          "message": "Improve coloring and API (#21)\n\n* Add LargestFirst ordering, column coloring, and star coloring\n\nRefactor greedy coloring into _greedy_color helper with LargestFirst\nvertex ordering (sort by decreasing degree) for fewer colors. Add\ncolor_cols for column coloring + JVP-based Jacobian computation, with\nauto direction selection in sparse_jacobian. Add star_color for\nsymmetric Hessian coloring (Gebremedhin et al. 2005) with symmetric\ndecompression, used by default in sparse_hessian.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add SparseMatrixColorings.jl attribution\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add SMC-sourced test cases for coloring algorithms\n\nPort test matrices from SparseMatrixColorings.jl: Gebremedhin et al.\nFigures 4.1 and 6.1, banded matrices with known star chromatic numbers,\nanti-diagonal, triangle, bidiagonal, and small hand-crafted patterns.\nTighten arrow matrix assertions to exact counts verified against SMC.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add unified color() function and ColoringResult dataclass\n\nReplace the separate colors/partition parameters on sparse_jacobian\nwith a single coloring: ColoringResult that carries the color array,\ncount, and partition together. The new color(sparsity) function\nauto-picks the best of row/column coloring (ties favor column/JVPs).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Rename ColoringResult to ColoredPattern and simplify sparse_jacobian API\n\nBundle the sparsity pattern into ColoredPattern so callers pass a single\nobject instead of separate sparsity and coloring arguments.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add coloring convenience functions, compressed visualization, and rename API\n\n- Add jacobian_coloring() and hessian_coloring() one-step convenience functions\n- Add ColoredPattern._compressed_pattern() and side-by-side/stacked __str__\n- Extract SparsityPattern._render() helper for reuse by ColoredPattern\n- Move ColoredPattern from coloring.py to pattern.py\n- Rename sparse_jacobian → jacobian, sparse_hessian → hessian\n- Add colored_pattern parameter to hessian() matching jacobian() API\n- Update README, CLAUDE.md, exports, tests, and pytest markers\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update README to describe automatic coloring and AD mode selection\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Rename color() to color_jacobian_pattern() and add color_hessian_pattern()\n\nThe generic name `color()` was misleading since it only handled Jacobians.\nThe new names clarify intent and `color_hessian_pattern` wraps star_color\nwith the same nnz==0 early-return guard, simplifying callers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix SIM108 lint warning and add test for size-0 binary elementwise\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix printing\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T16:52:31+01:00",
          "tree_id": "8f40488aafe282568526eaffc4c53420aa98cb6c",
          "url": "https://github.com/adrhill/asdex/commit/b9c2deaedc549dbf6297e04dac3dc07ea9c224e0"
        },
        "date": 1770479625958,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 734.8841250077743,
            "unit": "iter/sec",
            "range": "stddev: 0.002764833940172751",
            "extra": "mean: 1.3607587454544907 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3254.325339306064,
            "unit": "iter/sec",
            "range": "stddev: 0.000007946382942931392",
            "extra": "mean: 307.28335238087624 usec\nrounds: 1995"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 32.84250602193617,
            "unit": "iter/sec",
            "range": "stddev: 0.00043448072692210796",
            "extra": "mean: 30.448346400000048 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 49.2691786594981,
            "unit": "iter/sec",
            "range": "stddev: 0.014330025408853732",
            "extra": "mean: 20.296664714284216 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.284304365637617,
            "unit": "iter/sec",
            "range": "stddev: 0.01157854476877638",
            "extra": "mean: 46.982977823529296 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.42354962408476,
            "unit": "iter/sec",
            "range": "stddev: 0.0000316790361878196",
            "extra": "mean: 4.125011788461372 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.4969751006901695,
            "unit": "iter/sec",
            "range": "stddev: 0.0005464811702785462",
            "extra": "mean: 400.4845701999983 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 6.122329051911907,
            "unit": "iter/sec",
            "range": "stddev: 0.001691570595730755",
            "extra": "mean: 163.33653279999965 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 47.686912908530985,
            "unit": "iter/sec",
            "range": "stddev: 0.010106792125336104",
            "extra": "mean: 20.970114000000706 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.5666622322017,
            "unit": "iter/sec",
            "range": "stddev: 0.000429932547297812",
            "extra": "mean: 179.64086166666604 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.43661173848502344,
            "unit": "iter/sec",
            "range": "stddev: 0.00391796045611275",
            "extra": "mean: 2.290364440200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.275065790062376,
            "unit": "iter/sec",
            "range": "stddev: 0.015609194218410213",
            "extra": "mean: 3.635493893199995 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c77d21d1b704401c27825740687ee31cdff8831d",
          "message": "Pre-allocate decompression buffers (#22)\n\n* Pre-allocate decompression buffers and vectorize extraction\n\nCache BCOO indices, extraction index arrays, and seed matrices on\nColoredPattern so that repeated calls to jacobian/hessian with the\nsame colored pattern skip all redundant Python-side work.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add decompression benchmarks to dashboard\n\nAdd a `dashboard` marker for the 15 benchmarks tracked in the GitHub\nPages dashboard. Each group now covers detection, coloring,\ndecompression, materialization, and end-to-end.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T20:47:39+01:00",
          "tree_id": "ee398ac68726784184f03a22b5a067241e7fff23",
          "url": "https://github.com/adrhill/asdex/commit/c77d21d1b704401c27825740687ee31cdff8831d"
        },
        "date": 1770493735765,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 735.1467054168455,
            "unit": "iter/sec",
            "range": "stddev: 0.002595760710259541",
            "extra": "mean: 1.360272708333742 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3267.362485235797,
            "unit": "iter/sec",
            "range": "stddev: 0.000014058811323941864",
            "extra": "mean: 306.0572570440811 usec\nrounds: 2023"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_decompression",
            "value": 6971.952671046768,
            "unit": "iter/sec",
            "range": "stddev: 0.00004350945261481345",
            "extra": "mean: 143.43183999983466 usec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 33.48923308095324,
            "unit": "iter/sec",
            "range": "stddev: 0.00015061770290838248",
            "extra": "mean: 29.860343400002876 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 67.04596805661262,
            "unit": "iter/sec",
            "range": "stddev: 0.0005076461986959546",
            "extra": "mean: 14.915140000001413 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.473529459304483,
            "unit": "iter/sec",
            "range": "stddev: 0.014990568318235932",
            "extra": "mean: 46.56896305263408 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.11194672709567,
            "unit": "iter/sec",
            "range": "stddev: 0.000052866711172762246",
            "extra": "mean: 4.113331382774644 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_decompression",
            "value": 6480.901676642853,
            "unit": "iter/sec",
            "range": "stddev: 0.00002410616984129453",
            "extra": "mean: 154.2995172421758 usec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.5117968725381807,
            "unit": "iter/sec",
            "range": "stddev: 0.000647641836416642",
            "extra": "mean: 398.12136520000365 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 6.153978382959643,
            "unit": "iter/sec",
            "range": "stddev: 0.0011089291363504132",
            "extra": "mean: 162.49650839999674 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 45.400830694339376,
            "unit": "iter/sec",
            "range": "stddev: 0.011794530961575608",
            "extra": "mean: 22.02602870270127 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.3010490774356,
            "unit": "iter/sec",
            "range": "stddev: 0.0029035909641098585",
            "extra": "mean: 188.64190566667105 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_decompression",
            "value": 2468.50264676917,
            "unit": "iter/sec",
            "range": "stddev: 0.000045012300688880196",
            "extra": "mean: 405.10388000143394 usec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.43892968070805605,
            "unit": "iter/sec",
            "range": "stddev: 0.0184233671383794",
            "extra": "mean: 2.2782692626000083 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.2678616005525004,
            "unit": "iter/sec",
            "range": "stddev: 0.009598598028524339",
            "extra": "mean: 3.733271203999999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "89488df953a78b3644812d12be7ffffd3764b987",
          "message": "Batch AD calls via `vmap`, keep decompression on-device\n\nReplace Python for-loops over colors with jax.vmap for VJPs, JVPs,\nand HVPs, keeping all data on-device. Use JAX indexing instead of\nnumpy for decompression. Remove redundant vmap benchmarks from tests.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T21:33:21+01:00",
          "tree_id": "708c2f9402d225a4bbaea12b578c2e8242cb085b",
          "url": "https://github.com/adrhill/asdex/commit/89488df953a78b3644812d12be7ffffd3764b987"
        },
        "date": 1770496450932,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 727.301436412007,
            "unit": "iter/sec",
            "range": "stddev: 0.002791524460996081",
            "extra": "mean: 1.374945723926101 msec\nrounds: 163"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3261.790165840737,
            "unit": "iter/sec",
            "range": "stddev: 0.000007259324502826258",
            "extra": "mean: 306.58011372790037 usec\nrounds: 1996"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_decompression",
            "value": 1144.594309998927,
            "unit": "iter/sec",
            "range": "stddev: 0.00007391143476312659",
            "extra": "mean: 873.6719999952973 usec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 39.242004886886974,
            "unit": "iter/sec",
            "range": "stddev: 0.019707559042545512",
            "extra": "mean: 25.48289780000914 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.29673599131378,
            "unit": "iter/sec",
            "range": "stddev: 0.00046944607380491405",
            "extra": "mean: 12.151150199997574 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.94765248068138,
            "unit": "iter/sec",
            "range": "stddev: 0.014762750071992986",
            "extra": "mean: 45.56295945000102 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.01092003531514,
            "unit": "iter/sec",
            "range": "stddev: 0.00027559362983513004",
            "extra": "mean: 4.166477091345929 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_decompression",
            "value": 913.3014660748034,
            "unit": "iter/sec",
            "range": "stddev: 0.00010436475886961596",
            "extra": "mean: 1.0949287142807407 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.958610142897008,
            "unit": "iter/sec",
            "range": "stddev: 0.0004471235808667465",
            "extra": "mean: 33.37938560000566 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.062129139602666,
            "unit": "iter/sec",
            "range": "stddev: 0.026102539081607687",
            "extra": "mean: 99.38254480000523 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 45.78957217379344,
            "unit": "iter/sec",
            "range": "stddev: 0.013244675307261514",
            "extra": "mean: 21.839033485714157 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 4.951086520516684,
            "unit": "iter/sec",
            "range": "stddev: 0.0012975086037200852",
            "extra": "mean: 201.97586849999993 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_decompression",
            "value": 827.5514089760512,
            "unit": "iter/sec",
            "range": "stddev: 0.0002740368779836229",
            "extra": "mean: 1.2083841428501987 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.608438262115232,
            "unit": "iter/sec",
            "range": "stddev: 0.0009079043094998093",
            "extra": "mean: 42.357736199971896 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6443891470776115,
            "unit": "iter/sec",
            "range": "stddev: 0.023783692165777384",
            "extra": "mean: 1.5518572970000037 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "8c906fcdc43c840fcf4cc744c472dbfd11099bf6",
          "message": "Remove decompression benchmarks\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T21:43:35+01:00",
          "tree_id": "530d325a8edcda1b2859c554208833c7e43a9fa7",
          "url": "https://github.com/adrhill/asdex/commit/8c906fcdc43c840fcf4cc744c472dbfd11099bf6"
        },
        "date": 1770497062600,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 640.3495428431462,
            "unit": "iter/sec",
            "range": "stddev: 0.004719245327127077",
            "extra": "mean: 1.561647089744156 msec\nrounds: 156"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3240.6396371538676,
            "unit": "iter/sec",
            "range": "stddev: 0.000010220936174500636",
            "extra": "mean: 308.58105558391014 usec\nrounds: 1997"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.54359263898794,
            "unit": "iter/sec",
            "range": "stddev: 0.0008917314295607732",
            "extra": "mean: 17.68546979999428 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.01150183485781,
            "unit": "iter/sec",
            "range": "stddev: 0.0004783743952956105",
            "extra": "mean: 12.343926199991984 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.451508104309596,
            "unit": "iter/sec",
            "range": "stddev: 0.01227869954660014",
            "extra": "mean: 44.54043778947966 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.22197002117858,
            "unit": "iter/sec",
            "range": "stddev: 0.000039076235597581764",
            "extra": "mean: 4.111470686274455 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.638098661428163,
            "unit": "iter/sec",
            "range": "stddev: 0.0006911589741707278",
            "extra": "mean: 33.7403560000098 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.477285962148608,
            "unit": "iter/sec",
            "range": "stddev: 0.0017688366123536925",
            "extra": "mean: 87.12861239999938 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 50.173253031959945,
            "unit": "iter/sec",
            "range": "stddev: 0.0006052346434612794",
            "extra": "mean: 19.930938090918847 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.3672843601097595,
            "unit": "iter/sec",
            "range": "stddev: 0.002300898346681258",
            "extra": "mean: 186.31395933334716 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 17.40590769248556,
            "unit": "iter/sec",
            "range": "stddev: 0.03181374174194482",
            "extra": "mean: 57.45175819998849 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6563661491220518,
            "unit": "iter/sec",
            "range": "stddev: 0.011518020236517212",
            "extra": "mean: 1.523539873799996 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9f13bfd1303ff58eeb096d81748ddea232b82ecd",
          "message": "Support `while_loop`, `cond`, and `dynamic_slice` primitives (#24)\n\n* Add brusselator sparsity demo with diffrax\n\nDemonstrates asdex on a realistic reaction-diffusion ODE: detects the\nJacobian sparsity of the brusselator RHS (768 nnz, 11 colors) and\nshows the expected failure on diffrax's `while` primitive.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Support while_loop, cond, dynamic_slice, and diffrax primitives\n\nAdd handlers for 10 new JAX primitives needed by diffrax's diffeqsolve:\n- while: fixed-point iteration over body jaxpr\n- cond: union output deps across all branch jaxprs\n- dynamic_slice / dynamic_update_slice: precise when starts are static\n- not: zero derivative\n- select_if_vmap, nonbatchable, unvmap_any, unvmap_max, pure_callback:\n  conservative fallback\n\nAlso fix select_n to be element-wise instead of globally conservative,\nwhich is necessary for correct sparsity through diffrax control flow.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Simplify dynamic_slice handlers, move select_if_vmap to conservative\n\nReplace manual stride loops with np.indices + np.ravel_multi_index,\nextract _resolve_starts helper. Move select_if_vmap back to the\nconservative fallback since it's a different primitive from select_n.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix const_vals propagation into nested jaxprs, clean up handlers\n\nForward const_vals from outer-scope atoms to inner jaxpr variables so\nthat gather/scatter inside cond, while_loop, jit, and custom_jvp can\nresolve static indices precisely instead of falling back to conservative.\n\n- Add seed_const_vals and forward_const_vals helpers to _commons.py\n- Apply both helpers in prop_cond, prop_while, prop_nested_jaxpr,\n  and prop_custom_call\n- Extract prop_broadcast_in_dim into _broadcast.py\n- Clean up select_n: remove dead scalar broadcast, rename branches→cases\n- Add JAX doc URLs to handler docstrings\n- Add scan/associative_scan as explicit TODO errors\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Remove examples/ directory\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T00:17:07+01:00",
          "tree_id": "bb07835873f0a87d8bd6df37bd2c719aaf555793",
          "url": "https://github.com/adrhill/asdex/commit/9f13bfd1303ff58eeb096d81748ddea232b82ecd"
        },
        "date": 1770506274960,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 712.9003945119997,
            "unit": "iter/sec",
            "range": "stddev: 0.0029397769552276408",
            "extra": "mean: 1.4027205030297787 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3207.5571957888783,
            "unit": "iter/sec",
            "range": "stddev: 0.000007936247832828974",
            "extra": "mean: 311.7637313881339 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.33989375160917,
            "unit": "iter/sec",
            "range": "stddev: 0.0004206301673954495",
            "extra": "mean: 17.140929400002847 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.17760429766584,
            "unit": "iter/sec",
            "range": "stddev: 0.0003403211083427419",
            "extra": "mean: 12.472310800001196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.871985056219785,
            "unit": "iter/sec",
            "range": "stddev: 0.014465384747426549",
            "extra": "mean: 45.720587200000296 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.01367099820655,
            "unit": "iter/sec",
            "range": "stddev: 0.00003484183347459255",
            "extra": "mean: 4.114994830917887 msec\nrounds: 207"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.623636284809454,
            "unit": "iter/sec",
            "range": "stddev: 0.0002642951939543367",
            "extra": "mean: 34.936162200003196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.450401888038437,
            "unit": "iter/sec",
            "range": "stddev: 0.0009428151194036761",
            "extra": "mean: 87.33317919999308 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 47.654767640950446,
            "unit": "iter/sec",
            "range": "stddev: 0.010315826024813544",
            "extra": "mean: 20.984259277778644 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.0968249917397666,
            "unit": "iter/sec",
            "range": "stddev: 0.001562820144485054",
            "extra": "mean: 196.20057616666506 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 17.47956588057925,
            "unit": "iter/sec",
            "range": "stddev: 0.031907677120338165",
            "extra": "mean: 57.20965879999653 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6563157068829131,
            "unit": "iter/sec",
            "range": "stddev: 0.0072790151265666484",
            "extra": "mean: 1.5236569680000058 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4826cb6a0a83177816d1f47b643cd8478fecff5f",
          "message": "Clean up interpreter module and tests (#25)\n\n* Add JAX doc URLs and improve handler docstrings\n\n- Add JAX doc URLs to all handler docstrings missing them\n- Add JAX doc URLs to test module docstrings\n- Fix parameter names to match JAX API (start_indices, scatter_indices)\n- Document precise-path conditions in gather and scatter\n- Document conv assumption (feature_group_count=1, batch_group_count=1)\n- Clarify reduce_sum vs jax.lax.reduce naming difference\n- Document reshape bug with dimensions parameter\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix reshape handler ignoring `dimensions` parameter\n\nWhen `dimensions` is not None, `jax.lax.reshape` transposes the input\naxes before reshaping (e.g. `ravel(order='F')` emits `dimensions=(1,0)`).\nThe handler previously passed deps through in the original flat order,\nproducing incorrect (not merely conservative) sparsity patterns.\n\nFix by building a flat index mapping via `np.arange().reshape().transpose()`,\nmirroring the actual element reordering JAX performs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add 3D test for reshape dimensions fix\n\nTest ravel(order='F') on a (2, 3, 4) tensor,\nwhich emits dimensions=(2, 1, 0) — a higher-rank permutation\nthan the 2D case already tested.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Extract shared utilities from `_interpret` handlers\n\nAdd `atom_shape`, `flat_to_coords`, and `conservative_deps` helpers to\n`_commons.py` to reduce duplication across handler modules.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split `_indexing.py` into `_slice.py`, `_squeeze.py`, and `_reshape.py`\n\nEach handler now lives in its own module, consistent with the rest of\nthe `_interpret` package.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split test files to match `_interpret` handler modules\n\nOne test file per handler: `_foo.py` → `test_foo.py`.\nMove `test_control_flow.py` → `_interpret/test_select.py`,\nrename `test_dynamic_indexing.py` → `test_dynamic_slice.py`,\nand move fallback/custom_call tests into `test_internals.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T01:17:51+01:00",
          "tree_id": "bcca8d345906904d2f614bb3464bd4cc2c7d63e0",
          "url": "https://github.com/adrhill/asdex/commit/4826cb6a0a83177816d1f47b643cd8478fecff5f"
        },
        "date": 1770509917494,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 661.4605313115261,
            "unit": "iter/sec",
            "range": "stddev: 0.003868998829324764",
            "extra": "mean: 1.511806000000071 msec\nrounds: 160"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3274.813967175358,
            "unit": "iter/sec",
            "range": "stddev: 0.000007622603729831648",
            "extra": "mean: 305.3608571428365 usec\nrounds: 2058"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.08014669490991,
            "unit": "iter/sec",
            "range": "stddev: 0.0010245683445397706",
            "extra": "mean: 17.831622399995695 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 40.25880096439686,
            "unit": "iter/sec",
            "range": "stddev: 0.02635418907275455",
            "extra": "mean: 24.839289200002668 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.378545760733655,
            "unit": "iter/sec",
            "range": "stddev: 0.012485335515297359",
            "extra": "mean: 44.68565610526142 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.325166666264,
            "unit": "iter/sec",
            "range": "stddev: 0.000035022010391951884",
            "extra": "mean: 4.1097269702955295 msec\nrounds: 202"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.923406261884008,
            "unit": "iter/sec",
            "range": "stddev: 0.0010631299104126302",
            "extra": "mean: 34.57407440000679 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.759748849650636,
            "unit": "iter/sec",
            "range": "stddev: 0.02777464658779883",
            "extra": "mean: 102.46165299999461 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 43.62749628088578,
            "unit": "iter/sec",
            "range": "stddev: 0.013650826531622348",
            "extra": "mean: 22.921324514286262 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.30002260645173,
            "unit": "iter/sec",
            "range": "stddev: 0.0005517831193163254",
            "extra": "mean: 188.67844049998914 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.147545561808457,
            "unit": "iter/sec",
            "range": "stddev: 0.0002724181699510639",
            "extra": "mean: 43.20112460000587 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6490283559454185,
            "unit": "iter/sec",
            "range": "stddev: 0.008256847120334209",
            "extra": "mean: 1.5407647305999945 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6aadc3646bb475b13be22acad0a2816c4ed1ab28",
          "message": "Clean up public API and expand linting (#26)\n\n* Reorder modules for readability, rename `star_color` to `color_symmetric`\n\nPut public API functions at the top of coloring.py and decompression.py,\nwith private helpers below. Rename `star_color` to `color_symmetric` for\na consistent naming convention with `color_rows` and `color_cols`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Make docstrings consistent across jacobian/hessian function pairs\n\nStandardize f parameter descriptions, return value formatting,\nand cross-references across detection, coloring, and decompression.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update Claude settings\n\n* Simplify pattern.py: extract display, drop `nse` alias and `astype`\n\n- Move ~190 lines of visualization code (dots, braille, compressed\n  pattern rendering) from pattern.py to new _display.py module.\n  pattern.py goes from 561 to 321 lines.\n- Unify on `nnz` as the single property name, drop `nse` alias.\n- Remove `astype()` compatibility shim on SparsityPattern.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Expand ruff lint rules and fix all violations\n\nAdd PT, PIE, ICN, RSE, RET, D, PERF, T20, PLC0415 rules with Google\npydocstyle convention. Fix all violations: docstring formatting (D205,\nD403, D415), list comprehensions (PERF401), top-level imports\n(PLC0415), unnecessary else after return (RET505), pytest style (PT006,\nPT011), and ClassVar annotation (RUF012). Break circular import between\n_cond/_while and __init__ by passing prop_jaxpr as a callback.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T02:23:57+01:00",
          "tree_id": "1db64e11b9895efabdb7cc5203904e6ac38c04e2",
          "url": "https://github.com/adrhill/asdex/commit/6aadc3646bb475b13be22acad0a2816c4ed1ab28"
        },
        "date": 1770513887267,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 612.6945735212505,
            "unit": "iter/sec",
            "range": "stddev: 0.005448398002919871",
            "extra": "mean: 1.6321345793106103 msec\nrounds: 145"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3241.7554418963855,
            "unit": "iter/sec",
            "range": "stddev: 0.000009602478330656638",
            "extra": "mean: 308.4748426966511 usec\nrounds: 1958"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 54.8423190096478,
            "unit": "iter/sec",
            "range": "stddev: 0.0005226781676528589",
            "extra": "mean: 18.234094000001733 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 66.13110698290158,
            "unit": "iter/sec",
            "range": "stddev: 0.0005849134963231956",
            "extra": "mean: 15.121476799996003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.131987030739772,
            "unit": "iter/sec",
            "range": "stddev: 0.016345125512621825",
            "extra": "mean: 45.18347126315728 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 237.3843966652847,
            "unit": "iter/sec",
            "range": "stddev: 0.000039951660908164063",
            "extra": "mean: 4.212576791262376 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 19.25129184831375,
            "unit": "iter/sec",
            "range": "stddev: 0.03501476786651012",
            "extra": "mean: 51.944566000000236 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.815498025498261,
            "unit": "iter/sec",
            "range": "stddev: 0.0003944519437488283",
            "extra": "mean: 92.45991240000535 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 39.98685011264547,
            "unit": "iter/sec",
            "range": "stddev: 0.01887389307953793",
            "extra": "mean: 25.00822138235288 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.23865329894252,
            "unit": "iter/sec",
            "range": "stddev: 0.0019959349301302237",
            "extra": "mean: 190.88875383332984 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 21.769659827120982,
            "unit": "iter/sec",
            "range": "stddev: 0.0003420712859297489",
            "extra": "mean: 45.935490400000845 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6408715336879672,
            "unit": "iter/sec",
            "range": "stddev: 0.028526118680394126",
            "extra": "mean: 1.560375125799999 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "1cdf40724e1df3cab3eec9c7ce1136924c511b79",
          "message": "Update TODOs",
          "timestamp": "2026-02-08T02:31:33+01:00",
          "tree_id": "a559e6bf5951754308d220a50944ea8fd3227d0d",
          "url": "https://github.com/adrhill/asdex/commit/1cdf40724e1df3cab3eec9c7ce1136924c511b79"
        },
        "date": 1770514356131,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 712.6907432546981,
            "unit": "iter/sec",
            "range": "stddev: 0.002843062462979199",
            "extra": "mean: 1.4031331393939892 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3270.0731688782103,
            "unit": "iter/sec",
            "range": "stddev: 0.000015082466393729239",
            "extra": "mean: 305.80355495319003 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.54302540957044,
            "unit": "iter/sec",
            "range": "stddev: 0.0005413734758628894",
            "extra": "mean: 16.517179199999532 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.70956536121912,
            "unit": "iter/sec",
            "range": "stddev: 0.0003642976032704122",
            "extra": "mean: 12.09050000000218 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.288665869920912,
            "unit": "iter/sec",
            "range": "stddev: 0.013699066057989978",
            "extra": "mean: 46.97335221052606 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.684626458743,
            "unit": "iter/sec",
            "range": "stddev: 0.00003418447881757344",
            "extra": "mean: 4.103664701922856 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.617978520593457,
            "unit": "iter/sec",
            "range": "stddev: 0.0009777858479013306",
            "extra": "mean: 32.660549400000605 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.29548626238459,
            "unit": "iter/sec",
            "range": "stddev: 0.021716916662434395",
            "extra": "mean: 97.12994360000096 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 46.53926525566089,
            "unit": "iter/sec",
            "range": "stddev: 0.012149573734231483",
            "extra": "mean: 21.487232222222573 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.250599717592818,
            "unit": "iter/sec",
            "range": "stddev: 0.0016071691772904192",
            "extra": "mean: 190.45443450000002 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.091312666389808,
            "unit": "iter/sec",
            "range": "stddev: 0.0008468365036023959",
            "extra": "mean: 41.508738599998196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.645964836906171,
            "unit": "iter/sec",
            "range": "stddev: 0.022681246480375273",
            "extra": "mean: 1.5480718807999978 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "0ba34afc440e820d41974512e00e2abf9963bfa4",
          "message": "Update TODOs",
          "timestamp": "2026-02-08T02:34:14+01:00",
          "tree_id": "f53f6ae0e0efb95a0647e8b59b79e122e7846ecf",
          "url": "https://github.com/adrhill/asdex/commit/0ba34afc440e820d41974512e00e2abf9963bfa4"
        },
        "date": 1770514548007,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 694.1023545002508,
            "unit": "iter/sec",
            "range": "stddev: 0.003459001506109013",
            "extra": "mean: 1.4407097073168604 msec\nrounds: 164"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3283.288792643737,
            "unit": "iter/sec",
            "range": "stddev: 0.00000657511215361271",
            "extra": "mean: 304.5726596577543 usec\nrounds: 2045"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.28633186233281,
            "unit": "iter/sec",
            "range": "stddev: 0.00043161097943263897",
            "extra": "mean: 16.587507800002754 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.74478211876517,
            "unit": "iter/sec",
            "range": "stddev: 0.00024056347676489954",
            "extra": "mean: 12.384701199999881 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.435274383921353,
            "unit": "iter/sec",
            "range": "stddev: 0.01688006910089219",
            "extra": "mean: 46.65207368421196 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 244.54909670727938,
            "unit": "iter/sec",
            "range": "stddev: 0.000026353397777145783",
            "extra": "mean: 4.089158428571016 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.163248144096872,
            "unit": "iter/sec",
            "range": "stddev: 0.0008679081956083899",
            "extra": "mean: 33.15292819999911 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.21412332528969,
            "unit": "iter/sec",
            "range": "stddev: 0.0015875831170573263",
            "extra": "mean: 89.17326579999667 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 52.631467291941746,
            "unit": "iter/sec",
            "range": "stddev: 0.00042470736968479186",
            "extra": "mean: 19.00004030769454 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.380295879597647,
            "unit": "iter/sec",
            "range": "stddev: 0.0004908309553788828",
            "extra": "mean: 185.86338416666828 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.195135801886472,
            "unit": "iter/sec",
            "range": "stddev: 0.00027909114029105295",
            "extra": "mean: 43.11248739999485 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6386783892062186,
            "unit": "iter/sec",
            "range": "stddev: 0.013664416081184756",
            "extra": "mean: 1.5657332655999994 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "709b88779c03a2d3224f064d897e05c31776a249",
          "message": "Add MkDocs documentation site (#27)\n\n* Add MkDocs documentation site\n\nSet up MkDocs with Material theme following the Diataxis framework:\ntutorials, how-to guides, explanations, and auto-generated API reference.\nIncludes GitHub Actions workflow for deployment alongside existing benchmarks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Improve docs: executable code blocks, mermaid diagrams, content updates\n\n- Add markdown-exec for live code execution in docs\n- Add mermaid diagrams to pipeline explanation page\n- Rewrite getting-started with one-call API, benchmarks, and Hessian example\n- Rename \"The 3-Stage Pipeline\" to \"The ASD Pipeline\"\n- Remove contributing page (user edit)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Restructure docs: add Tutorials tab, executable examples, and manual sparsity docs\n\n- Move getting-started.md into tutorials/ folder with its own nav tab\n- Remove section overview index.md pages (tutorials, how-to, explanation)\n- Add executable code blocks with hidden print cells to how-to guides\n- Document manual SparsityPattern construction (from_dense, from_coordinates, from_bcoo)\n- Add precompute warning admonitions to one-call API sections\n- Expand docs/CLAUDE.md with structure overview, clearer Diataxis descriptions, and restart note\n- Update mkdocs.yml nav and fix cross-section links\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Improve Hessian how-to guide structure, gitignore site/, and polish tutorial\n\n- Restructure hessians.md to match jacobians.md layout (separate sections for\n  manual sparsity, detect+color, precomputing)\n- Add executable code blocks with hidden prints to hessians.md\n- Use sparse multi-dimensional example (sum(x**3)) instead of dense one\n- Add site/ to .gitignore\n- Minor tutorial polish\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split API reference into per-topic pages and remove pipeline explanation\n\n- Remove explanation/pipeline.md\n- Split reference/api.md into jacobian.md, hessian.md, sparsity.md,\n  coloring.md, and data-structures.md\n- Keep reference/api.md as a full API page with all docstrings\n- Update nav, docs/CLAUDE.md structure tree, and cross-links\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add custom CSS for sidebar section headers and tweak reference pages\n\n- Style sidebar section headers as small, bold, uppercase labels\n  to visually distinguish them from page links\n- Move color_jacobian_pattern/color_hessian_pattern back to coloring reference\n- Swap order of data structures (ColoredPattern first)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add sparsity detection explanation page\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update the Getting started guide.\n\n* Trim README\n\n* Fix API reference\n\n* Expand the graph coloring explanation page\n\nAdd concrete Jacobian example, compression/decompression section,\nand references. Streamline symmetric coloring discussion.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T16:55:30+01:00",
          "tree_id": "0579cbee5577e6e920b7f81d5bc2f01c23d7bba1",
          "url": "https://github.com/adrhill/asdex/commit/709b88779c03a2d3224f064d897e05c31776a249"
        },
        "date": 1770566178780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 723.1767987559738,
            "unit": "iter/sec",
            "range": "stddev: 0.002757190643072523",
            "extra": "mean: 1.3827877245512081 msec\nrounds: 167"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3198.8332419460407,
            "unit": "iter/sec",
            "range": "stddev: 0.000007690668532322336",
            "extra": "mean: 312.6139827756825 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.48627350301963,
            "unit": "iter/sec",
            "range": "stddev: 0.00045187093142807697",
            "extra": "mean: 16.263792600000215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.36331633943327,
            "unit": "iter/sec",
            "range": "stddev: 0.0003361716729393651",
            "extra": "mean: 11.99568400000146 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.444171310600588,
            "unit": "iter/sec",
            "range": "stddev: 0.012198139784861858",
            "extra": "mean: 44.554997649999706 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.40403633936577,
            "unit": "iter/sec",
            "range": "stddev: 0.000024446120423793942",
            "extra": "mean: 4.177039014423533 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.961574741357943,
            "unit": "iter/sec",
            "range": "stddev: 0.0010937023342586971",
            "extra": "mean: 32.298098800001185 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.507784520242707,
            "unit": "iter/sec",
            "range": "stddev: 0.0015505797816357453",
            "extra": "mean: 86.89769939999792 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 46.879095267421185,
            "unit": "iter/sec",
            "range": "stddev: 0.011706003566131759",
            "extra": "mean: 21.331469694445104 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.0739246208727735,
            "unit": "iter/sec",
            "range": "stddev: 0.0006657095718346416",
            "extra": "mean: 197.08609699999613 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.886192156990347,
            "unit": "iter/sec",
            "range": "stddev: 0.0006941844016833498",
            "extra": "mean: 43.694468400002506 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6523400235172981,
            "unit": "iter/sec",
            "range": "stddev: 0.012243624098424461",
            "extra": "mean: 1.5329428886000016 sec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f4fffeb04244ff84d63c4d73800a96aaef9dd657",
          "message": "Add precise `pad` primitive handler (#28)\n\n* Add precise `pad` primitive handler\n\nReplaces the conservative fallback with a handler that reverse-maps each\noutput element to its input source, correctly handling low/high padding,\nnegative padding (trimming), and interior padding (dilation).\n\nThis fixes Hessian sparsity for functions using sliced operations\n(e.g. finite differences), where JAX's grad emits `pad` primitives.\nFor example, `sum((x[1:]-x[:-1])^2)` now gives the correct 13-nnz\ntridiagonal Hessian instead of a fully dense 25-nnz pattern.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update project metadata in pyproject.toml\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T20:43:50+01:00",
          "tree_id": "e8a3cd6e5d415c5d0462305804b0690d222e2990",
          "url": "https://github.com/adrhill/asdex/commit/f4fffeb04244ff84d63c4d73800a96aaef9dd657"
        },
        "date": 1770579863461,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 697.5422604595892,
            "unit": "iter/sec",
            "range": "stddev: 0.0032393256818053326",
            "extra": "mean: 1.4336048963415216 msec\nrounds: 164"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3233.9243486365335,
            "unit": "iter/sec",
            "range": "stddev: 0.000006546585548055818",
            "extra": "mean: 309.22182840226725 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.95912993834298,
            "unit": "iter/sec",
            "range": "stddev: 0.000497500155944088",
            "extra": "mean: 16.67802719999969 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.12605956108412,
            "unit": "iter/sec",
            "range": "stddev: 0.0003830124252539208",
            "extra": "mean: 12.326495400002102 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.550483462811844,
            "unit": "iter/sec",
            "range": "stddev: 0.008643191556736669",
            "extra": "mean: 42.461973299999656 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.95387575596135,
            "unit": "iter/sec",
            "range": "stddev: 0.000026421486727964214",
            "extra": "mean: 4.116007603864961 msec\nrounds: 207"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 23.93998739281899,
            "unit": "iter/sec",
            "range": "stddev: 0.019302575564457542",
            "extra": "mean: 41.77111640000106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.273706392078129,
            "unit": "iter/sec",
            "range": "stddev: 0.0023934340812666946",
            "extra": "mean: 88.70197299999631 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.83794533076049,
            "unit": "iter/sec",
            "range": "stddev: 0.012183160993606203",
            "extra": "mean: 11.256451241379015 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3238.915607738146,
            "unit": "iter/sec",
            "range": "stddev: 0.000009191702282507735",
            "extra": "mean: 308.74530895799927 usec\nrounds: 2188"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.792111820396922,
            "unit": "iter/sec",
            "range": "stddev: 0.0014094636234731404",
            "extra": "mean: 40.335410200000865 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.012303892316236,
            "unit": "iter/sec",
            "range": "stddev: 0.011866928769555456",
            "extra": "mean: 52.597518199998206 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "e3ae944fb70fbc30c69a3de0300327d88b7c8f96",
          "message": "Docs: use Rosenbrock example in Hessian How-to\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T20:54:59+01:00",
          "tree_id": "814807c3bf5f9c460272cf4085dc0d754a21af49",
          "url": "https://github.com/adrhill/asdex/commit/e3ae944fb70fbc30c69a3de0300327d88b7c8f96"
        },
        "date": 1770580538117,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 656.8824266643404,
            "unit": "iter/sec",
            "range": "stddev: 0.004356284314289112",
            "extra": "mean: 1.5223424457829025 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3235.0562435182023,
            "unit": "iter/sec",
            "range": "stddev: 0.000011401429372377512",
            "extra": "mean: 309.11363658780647 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.77068808160453,
            "unit": "iter/sec",
            "range": "stddev: 0.0005521331337005912",
            "extra": "mean: 16.730608799997526 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.23589298100775,
            "unit": "iter/sec",
            "range": "stddev: 0.00032859974012539435",
            "extra": "mean: 12.309829599999489 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.184254099399144,
            "unit": "iter/sec",
            "range": "stddev: 0.015734912948974146",
            "extra": "mean: 47.20487185000124 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.86002415603278,
            "unit": "iter/sec",
            "range": "stddev: 0.00009739204979430773",
            "extra": "mean: 4.151789004854475 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.560585910674885,
            "unit": "iter/sec",
            "range": "stddev: 0.0009355068356940787",
            "extra": "mean: 33.82882880000295 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.398375503523575,
            "unit": "iter/sec",
            "range": "stddev: 0.0008333071314144641",
            "extra": "mean: 87.7317999999974 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 89.21798401914732,
            "unit": "iter/sec",
            "range": "stddev: 0.012858542948533544",
            "extra": "mean: 11.208502534482141 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3193.6488062075427,
            "unit": "iter/sec",
            "range": "stddev: 0.000007983126948381369",
            "extra": "mean: 313.12146722466326 usec\nrounds: 2151"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.20606944810584,
            "unit": "iter/sec",
            "range": "stddev: 0.0006486529175009562",
            "extra": "mean: 39.67298439999922 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.766603159825255,
            "unit": "iter/sec",
            "range": "stddev: 0.012681069918010959",
            "extra": "mean: 53.28614834999854 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2fad0d1e9176f840f75b880c962f6d8e5c38375f",
          "message": "Add precise `transpose` primitive handler (#29)\n\n* Add precise `transpose` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add tests for `transpose` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add `/add-handler` skill for new primitive handlers\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T21:48:54+01:00",
          "tree_id": "643ff1683ef75e3d8e7d4d3fa50b8214b01c5317",
          "url": "https://github.com/adrhill/asdex/commit/2fad0d1e9176f840f75b880c962f6d8e5c38375f"
        },
        "date": 1770583770925,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 627.2783058650452,
            "unit": "iter/sec",
            "range": "stddev: 0.0046852126152313045",
            "extra": "mean: 1.5941887207799328 msec\nrounds: 154"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3236.4013890556494,
            "unit": "iter/sec",
            "range": "stddev: 0.00000887699830795827",
            "extra": "mean: 308.98515968434634 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.17520850940324,
            "unit": "iter/sec",
            "range": "stddev: 0.00044086822577100993",
            "extra": "mean: 17.490098000001808 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 79.45349092184229,
            "unit": "iter/sec",
            "range": "stddev: 0.0007122958802572632",
            "extra": "mean: 12.58597939999504 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.873501365547583,
            "unit": "iter/sec",
            "range": "stddev: 0.018059539214148643",
            "extra": "mean: 47.90763094736629 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.7553895811917,
            "unit": "iter/sec",
            "range": "stddev: 0.000024419997101398102",
            "extra": "mean: 4.136412436274384 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 20.319049223247458,
            "unit": "iter/sec",
            "range": "stddev: 0.03233823186424366",
            "extra": "mean: 49.21490119999703 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.011243221486124,
            "unit": "iter/sec",
            "range": "stddev: 0.00037835694302140857",
            "extra": "mean: 90.81626660000666 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 81.09136295837334,
            "unit": "iter/sec",
            "range": "stddev: 0.01634039089959695",
            "extra": "mean: 12.33176954385796 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3226.416886307873,
            "unit": "iter/sec",
            "range": "stddev: 0.000010074788975693353",
            "extra": "mean: 309.9413483247488 usec\nrounds: 2090"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.456390565154894,
            "unit": "iter/sec",
            "range": "stddev: 0.0013220869302057214",
            "extra": "mean: 40.889108199996826 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.947357431648637,
            "unit": "iter/sec",
            "range": "stddev: 0.016950621194069877",
            "extra": "mean: 55.71850919047198 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d1b7e377204a3c50537d848b7f8411b4c40a0819",
          "message": "Add precise `rev` primitive handler (#30)\n\n* Add precise `rev` primitive handler\n\nReplace the conservative fallback for `rev` (reverse) with a precise\nhandler that tracks the per-element permutation along reversed dimensions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Vectorize `prop_rev` and improve add-handler skill\n\nReplace per-element Python loop in `_rev.py` with `np.flip` on an\nindex array.  Update the add-handler skill with detailed adversarial\ntest guidance (step 7) and a new simplification step (step 8).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T22:06:35+01:00",
          "tree_id": "8f6ed223b04ee6c4a5d6578cfd68d8c69b7544c2",
          "url": "https://github.com/adrhill/asdex/commit/d1b7e377204a3c50537d848b7f8411b4c40a0819"
        },
        "date": 1770584829554,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 688.062072190994,
            "unit": "iter/sec",
            "range": "stddev: 0.003341266436772559",
            "extra": "mean: 1.4533572484466453 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3266.6485602061334,
            "unit": "iter/sec",
            "range": "stddev: 0.000006893697830328382",
            "extra": "mean: 306.1241457626827 usec\nrounds: 2065"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.47664864153817,
            "unit": "iter/sec",
            "range": "stddev: 0.0005048331997152025",
            "extra": "mean: 16.53530779999528 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.39657243590915,
            "unit": "iter/sec",
            "range": "stddev: 0.0006694529387191353",
            "extra": "mean: 12.285529600001155 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.817267270073398,
            "unit": "iter/sec",
            "range": "stddev: 0.014135430470461764",
            "extra": "mean: 45.8352545999972 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 237.88885136984004,
            "unit": "iter/sec",
            "range": "stddev: 0.000024086860476311063",
            "extra": "mean: 4.203643820387884 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.84008990873228,
            "unit": "iter/sec",
            "range": "stddev: 0.0009300327449467559",
            "extra": "mean: 32.42532699999856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.40471622925195,
            "unit": "iter/sec",
            "range": "stddev: 0.0010717385893134485",
            "extra": "mean: 87.68302340001242 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.96543990937641,
            "unit": "iter/sec",
            "range": "stddev: 0.011526973949098582",
            "extra": "mean: 10.64221059321848 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3207.7035312665535,
            "unit": "iter/sec",
            "range": "stddev: 0.000006946108648670315",
            "extra": "mean: 311.7495087225697 usec\nrounds: 2121"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.970894799000565,
            "unit": "iter/sec",
            "range": "stddev: 0.0013706872503722798",
            "extra": "mean: 40.046622600004866 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.722099998943648,
            "unit": "iter/sec",
            "range": "stddev: 0.01449612935707989",
            "extra": "mean: 53.41281160000335 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7a3887d837973c22afaf1641e1f3b36201bb8cb7",
          "message": "Add precise `dot_general` primitive handler (#31)\n\n* Add precise `dot_general` primitive handler\n\nReplace the conservative fallback for dot_general with a precise handler\nthat tracks per-element dependencies through batch, contracting, and free\ndimensions of generalized matrix multiplications.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Clean up TODOs\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T22:24:53+01:00",
          "tree_id": "7c738b1db11b337fff38cf52f921d98889c167a4",
          "url": "https://github.com/adrhill/asdex/commit/7a3887d837973c22afaf1641e1f3b36201bb8cb7"
        },
        "date": 1770585929537,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 721.949824419193,
            "unit": "iter/sec",
            "range": "stddev: 0.0028176224725377542",
            "extra": "mean: 1.3851378117648243 msec\nrounds: 170"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3254.8889196749687,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672363262213771",
            "extra": "mean: 307.23014661276346 usec\nrounds: 1978"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.08618170461724,
            "unit": "iter/sec",
            "range": "stddev: 0.00044032042771823915",
            "extra": "mean: 16.370314399998165 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.20000947813111,
            "unit": "iter/sec",
            "range": "stddev: 0.00026824467831327475",
            "extra": "mean: 12.019229400002018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.1320355054603,
            "unit": "iter/sec",
            "range": "stddev: 0.013144622159524619",
            "extra": "mean: 45.18337229999858 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.97785113300526,
            "unit": "iter/sec",
            "range": "stddev: 0.000029756373135880102",
            "extra": "mean: 4.098732714285802 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.06508430901801,
            "unit": "iter/sec",
            "range": "stddev: 0.0007681014319213438",
            "extra": "mean: 32.190480800005616 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.544441683578457,
            "unit": "iter/sec",
            "range": "stddev: 0.0012331872676036783",
            "extra": "mean: 86.62177239999949 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 97.2132077878127,
            "unit": "iter/sec",
            "range": "stddev: 0.010822953475703495",
            "extra": "mean: 10.286668064515473 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3147.8071043122823,
            "unit": "iter/sec",
            "range": "stddev: 0.00002140317040580055",
            "extra": "mean: 317.6814737567839 usec\nrounds: 2172"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.614666047944915,
            "unit": "iter/sec",
            "range": "stddev: 0.0012183995004755167",
            "extra": "mean: 39.04013420000183 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.05463142851476,
            "unit": "iter/sec",
            "range": "stddev: 0.01258777165146568",
            "extra": "mean: 52.48067923809464 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b8bca5034576ff130f6f22a982caf095f01de617",
          "message": "Add thorough tests for `reshape` primitive handler (#32)\n\nExpand test_reshape.py from 4 to 25 tests covering identity reshapes,\ndimensions param permutations, size-1 dims, chained reshapes,\nnon-contiguous inputs, high-level functions, and compositions.\nUpdate TODO.md to remove outdated entries.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T23:14:22+01:00",
          "tree_id": "a93714cb8e1b49cf248350fb5f84b416f4362fcc",
          "url": "https://github.com/adrhill/asdex/commit/b8bca5034576ff130f6f22a982caf095f01de617"
        },
        "date": 1770588894705,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 706.6814728854604,
            "unit": "iter/sec",
            "range": "stddev: 0.00275834972261733",
            "extra": "mean: 1.41506469090931 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3210.5284238577497,
            "unit": "iter/sec",
            "range": "stddev: 0.00001555082009860901",
            "extra": "mean: 311.47520531788547 usec\nrounds: 2031"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.8414837394148,
            "unit": "iter/sec",
            "range": "stddev: 0.0005442975279964836",
            "extra": "mean: 16.710815600004025 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.36858708735289,
            "unit": "iter/sec",
            "range": "stddev: 0.00039097645122457096",
            "extra": "mean: 12.28975500000331 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.99785972171242,
            "unit": "iter/sec",
            "range": "stddev: 0.014321652543305398",
            "extra": "mean: 45.45896794736698 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.36885609068517,
            "unit": "iter/sec",
            "range": "stddev: 0.000025899599836146022",
            "extra": "mean: 4.1776529174754 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.294610604526806,
            "unit": "iter/sec",
            "range": "stddev: 0.0010932768675020377",
            "extra": "mean: 33.00917159999983 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.299659591427007,
            "unit": "iter/sec",
            "range": "stddev: 0.000491268843826839",
            "extra": "mean: 88.49824119999994 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 125.73370669657999,
            "unit": "iter/sec",
            "range": "stddev: 0.00016509435429521685",
            "extra": "mean: 7.953316785714395 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3218.342954701133,
            "unit": "iter/sec",
            "range": "stddev: 0.000011137192121113374",
            "extra": "mean: 310.7189053731111 usec\nrounds: 2103"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.039462067021642,
            "unit": "iter/sec",
            "range": "stddev: 0.0009776746083524448",
            "extra": "mean: 39.936960199997884 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.915778143726822,
            "unit": "iter/sec",
            "range": "stddev: 0.01666804526993325",
            "extra": "mean: 55.81672155000135 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "faef7a288b68572f0fd1c95936b834140967e993",
          "message": "Add precise handlers for all reduction primitives (#33)\n\n* Add precise `reduce_max` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add precise `reduce_prod` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add precise `reduce_min` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add zero-derivative handlers for `reduce_and`, `reduce_or`, `reduce_xor`\n\nBitwise reductions have zero Jacobian, so they use the existing\n`prop_zero_derivative` — no separate handler files needed.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Consolidate reduction handlers into single `_reduce.py` module\n\nThe four reduction primitives (reduce_sum, reduce_max, reduce_min,\nreduce_prod) share identical sparsity structure, so they now share\na single `prop_reduce` handler. Tests are parametrized over the\nreduce function and shape/axes combinations.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Parametrize bitwise reduction tests over reduce_and/or/xor\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update SKILL and TODOs\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T23:23:59+01:00",
          "tree_id": "984e3a6736257a3c4644e001b388e6e3c720fbd6",
          "url": "https://github.com/adrhill/asdex/commit/faef7a288b68572f0fd1c95936b834140967e993"
        },
        "date": 1770589473351,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 663.8023996695964,
            "unit": "iter/sec",
            "range": "stddev: 0.00405260089833369",
            "extra": "mean: 1.5064724088038004 msec\nrounds: 159"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3239.075439673935,
            "unit": "iter/sec",
            "range": "stddev: 0.0000065504970684697765",
            "extra": "mean: 308.73007394377515 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.2030569862779,
            "unit": "iter/sec",
            "range": "stddev: 0.0007613997673490583",
            "extra": "mean: 17.481583200000728 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 77.37671038705516,
            "unit": "iter/sec",
            "range": "stddev: 0.00026495798700692825",
            "extra": "mean: 12.923785399996746 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.776023151863665,
            "unit": "iter/sec",
            "range": "stddev: 0.019496801653107812",
            "extra": "mean: 48.13240689473805 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.95915897840848,
            "unit": "iter/sec",
            "range": "stddev: 0.000035858621969400084",
            "extra": "mean: 4.115918099999963 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.72498049889885,
            "unit": "iter/sec",
            "range": "stddev: 0.0007768260677243277",
            "extra": "mean: 34.812904399998956 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.085312804569499,
            "unit": "iter/sec",
            "range": "stddev: 0.0009905455258238083",
            "extra": "mean: 90.20945260000133 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 123.79472041720724,
            "unit": "iter/sec",
            "range": "stddev: 0.00018826480371296914",
            "extra": "mean: 8.077888916666609 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3102.2420231733677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000534793497042519",
            "extra": "mean: 322.34751271181375 usec\nrounds: 2124"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.786655063206233,
            "unit": "iter/sec",
            "range": "stddev: 0.0017211592353249784",
            "extra": "mean: 40.34429000000159 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.42817061122612,
            "unit": "iter/sec",
            "range": "stddev: 0.001038009636240581",
            "extra": "mean: 51.47165011111099 msec\nrounds: 9"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "c55805a539c8645fcdbcc0c7aebd74e29162b6fb",
          "message": "Update README",
          "timestamp": "2026-02-08T23:45:23+01:00",
          "tree_id": "3cb74fbb0ca19e3325dfdab84a6ec1e2226da9df",
          "url": "https://github.com/adrhill/asdex/commit/c55805a539c8645fcdbcc0c7aebd74e29162b6fb"
        },
        "date": 1770590756500,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 724.5264163270951,
            "unit": "iter/sec",
            "range": "stddev: 0.0033559206792397305",
            "extra": "mean: 1.3802119252868474 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3417.4198073177645,
            "unit": "iter/sec",
            "range": "stddev: 0.0002142352700067656",
            "extra": "mean: 292.6184245373329 usec\nrounds: 2054"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 65.2019058516735,
            "unit": "iter/sec",
            "range": "stddev: 0.0004551153652242098",
            "extra": "mean: 15.336975000008124 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.53170017842771,
            "unit": "iter/sec",
            "range": "stddev: 0.0019781483005266347",
            "extra": "mean: 12.11655640000231 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.68647945823764,
            "unit": "iter/sec",
            "range": "stddev: 0.0014967385600157062",
            "extra": "mean: 40.50800365000242 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 264.6384066970842,
            "unit": "iter/sec",
            "range": "stddev: 0.000024586418878405413",
            "extra": "mean: 3.778741009216551 msec\nrounds: 217"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.64875532155554,
            "unit": "iter/sec",
            "range": "stddev: 0.0008522561011637792",
            "extra": "mean: 32.627752400003374 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 12.497271783085472,
            "unit": "iter/sec",
            "range": "stddev: 0.0009078759042344923",
            "extra": "mean: 80.0174643999867 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.22536269465958,
            "unit": "iter/sec",
            "range": "stddev: 0.013022589038251285",
            "extra": "mean: 10.843004253730154 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3460.943119870303,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043635505857461",
            "extra": "mean: 288.938582740266 usec\nrounds: 2248"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.085065171548845,
            "unit": "iter/sec",
            "range": "stddev: 0.0005627823384392886",
            "extra": "mean: 34.38190679999593 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 21.410205847082395,
            "unit": "iter/sec",
            "range": "stddev: 0.013586233194707512",
            "extra": "mean: 46.70669713043752 msec\nrounds: 23"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4bd231ae9d40e1bc5f2e6bca63cdda8b672d02dc",
          "message": "docs: add Brusselator PDE how-to page (#34)\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-09T13:36:13+01:00",
          "tree_id": "b8b8b773bd5f7537740a8489ca504621aee8c29d",
          "url": "https://github.com/adrhill/asdex/commit/4bd231ae9d40e1bc5f2e6bca63cdda8b672d02dc"
        },
        "date": 1770640607310,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 702.9768406410767,
            "unit": "iter/sec",
            "range": "stddev: 0.0030694296715492557",
            "extra": "mean: 1.4225219696968314 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3265.929058375829,
            "unit": "iter/sec",
            "range": "stddev: 0.000011803480736036732",
            "extra": "mean: 306.1915865671949 usec\nrounds: 2010"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.892165698679335,
            "unit": "iter/sec",
            "range": "stddev: 0.0006949599381724919",
            "extra": "mean: 17.273494399999834 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.58182008063119,
            "unit": "iter/sec",
            "range": "stddev: 0.0004395750268183218",
            "extra": "mean: 12.409746999998106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.873975457167514,
            "unit": "iter/sec",
            "range": "stddev: 0.01386214872957634",
            "extra": "mean: 45.71642690000033 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 224.18660175246134,
            "unit": "iter/sec",
            "range": "stddev: 0.0008591072551944504",
            "extra": "mean: 4.460569865384567 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.05665770146733,
            "unit": "iter/sec",
            "range": "stddev: 0.001011004708438243",
            "extra": "mean: 33.27049899999963 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.263280432588632,
            "unit": "iter/sec",
            "range": "stddev: 0.001202995618289872",
            "extra": "mean: 88.78408079999929 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 94.57867191689662,
            "unit": "iter/sec",
            "range": "stddev: 0.011294974110691617",
            "extra": "mean: 10.573208311475014 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3224.2280377520324,
            "unit": "iter/sec",
            "range": "stddev: 0.00000966464352875787",
            "extra": "mean: 310.1517598293733 usec\nrounds: 2111"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.8706951919908,
            "unit": "iter/sec",
            "range": "stddev: 0.0009797297084456064",
            "extra": "mean: 38.653773800001545 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.61011801170359,
            "unit": "iter/sec",
            "range": "stddev: 0.013845812577498744",
            "extra": "mean: 53.734210571427695 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fdd1c6c2527b11ed5fb8fe0eb25e95486c790655",
          "message": "feat!: switch to functional API (#35)\n\n* feat!: switch to functional API\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: rename returned closures to jac_fn / hess_fn\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Minor tweaks\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-09T16:39:01+01:00",
          "tree_id": "f5a687e1310b7a3e6232ceb990a5b4c331bed0af",
          "url": "https://github.com/adrhill/asdex/commit/fdd1c6c2527b11ed5fb8fe0eb25e95486c790655"
        },
        "date": 1770651579278,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 690.6907152366679,
            "unit": "iter/sec",
            "range": "stddev: 0.0034354438004304878",
            "extra": "mean: 1.447826035503237 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3250.5414913495024,
            "unit": "iter/sec",
            "range": "stddev: 0.000007300839078000627",
            "extra": "mean: 307.6410507791542 usec\nrounds: 1989"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.15583525679842,
            "unit": "iter/sec",
            "range": "stddev: 0.0004754775896584963",
            "extra": "mean: 16.35166939999948 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.8207392141831,
            "unit": "iter/sec",
            "range": "stddev: 0.00032558572711124777",
            "extra": "mean: 12.074270400000842 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.391676110269447,
            "unit": "iter/sec",
            "range": "stddev: 0.000397813980269722",
            "extra": "mean: 40.99759260000084 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.41475115351932,
            "unit": "iter/sec",
            "range": "stddev: 0.00003322362381768263",
            "extra": "mean: 4.1422489521532375 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.54673530452859,
            "unit": "iter/sec",
            "range": "stddev: 0.0007840246433365911",
            "extra": "mean: 32.73672260000069 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.64007868218244,
            "unit": "iter/sec",
            "range": "stddev: 0.000969405930030996",
            "extra": "mean: 85.91007220000222 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.74963893363639,
            "unit": "iter/sec",
            "range": "stddev: 0.013051003496385981",
            "extra": "mean: 10.78171313114775 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3187.335942598409,
            "unit": "iter/sec",
            "range": "stddev: 0.000008791022522623594",
            "extra": "mean: 313.74163816091846 usec\nrounds: 2175"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.066211534625285,
            "unit": "iter/sec",
            "range": "stddev: 0.0014928111163875154",
            "extra": "mean: 38.36384120000105 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.76281618707656,
            "unit": "iter/sec",
            "range": "stddev: 0.01431221401884499",
            "extra": "mean: 50.60007594737065 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "3f8fad1869dc0d6cca9d4b79891f6f7d4bb6ee6d",
          "message": "docs: add acknowledgements",
          "timestamp": "2026-02-09T17:01:20+01:00",
          "tree_id": "69e931f164cf4341aae0fe755fadd352ad3b645b",
          "url": "https://github.com/adrhill/asdex/commit/3f8fad1869dc0d6cca9d4b79891f6f7d4bb6ee6d"
        },
        "date": 1770652915672,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 690.7146165524471,
            "unit": "iter/sec",
            "range": "stddev: 0.0033168544132346897",
            "extra": "mean: 1.4477759352933404 msec\nrounds: 170"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3255.1562447542483,
            "unit": "iter/sec",
            "range": "stddev: 0.000009086257618802929",
            "extra": "mean: 307.20491577371155 usec\nrounds: 1959"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.78769961829452,
            "unit": "iter/sec",
            "range": "stddev: 0.00023672338029550737",
            "extra": "mean: 16.725848399994447 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.11863718937268,
            "unit": "iter/sec",
            "range": "stddev: 0.00030809339140293074",
            "extra": "mean: 12.177503599991724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.666926401011754,
            "unit": "iter/sec",
            "range": "stddev: 0.011099801976726564",
            "extra": "mean: 44.11714152631493 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.8267499046422,
            "unit": "iter/sec",
            "range": "stddev: 0.00004275722683018239",
            "extra": "mean: 4.1696766536577385 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 21.31054480279828,
            "unit": "iter/sec",
            "range": "stddev: 0.027003823010725823",
            "extra": "mean: 46.92512600000214 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.210848801165703,
            "unit": "iter/sec",
            "range": "stddev: 0.000568256521969903",
            "extra": "mean: 89.19931199999951 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 83.74341947148179,
            "unit": "iter/sec",
            "range": "stddev: 0.01510216505124126",
            "extra": "mean: 11.941236771929796 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3232.8468006097764,
            "unit": "iter/sec",
            "range": "stddev: 0.000011943892223007663",
            "extra": "mean: 309.3248958816672 usec\nrounds: 2161"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.57539541098198,
            "unit": "iter/sec",
            "range": "stddev: 0.0011113620270840575",
            "extra": "mean: 40.691105200005495 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.97252283889614,
            "unit": "iter/sec",
            "range": "stddev: 0.014506039165353698",
            "extra": "mean: 52.70780319999773 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "023d1c9221e867e4f619ad54c3ba5214a7a2b590",
          "message": "docs: link to GitHub profiles",
          "timestamp": "2026-02-09T17:07:01+01:00",
          "tree_id": "7b7081bad612bd3d8c72349ec77dbb0391c644b6",
          "url": "https://github.com/adrhill/asdex/commit/023d1c9221e867e4f619ad54c3ba5214a7a2b590"
        },
        "date": 1770653266598,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 673.927732539104,
            "unit": "iter/sec",
            "range": "stddev: 0.003648992710532731",
            "extra": "mean: 1.4838386249997153 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3179.0926706922714,
            "unit": "iter/sec",
            "range": "stddev: 0.00004418683749074385",
            "extra": "mean: 314.55515884104204 usec\nrounds: 2002"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.34550764695828,
            "unit": "iter/sec",
            "range": "stddev: 0.002226169274872178",
            "extra": "mean: 17.438157600005866 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.29554532114807,
            "unit": "iter/sec",
            "range": "stddev: 0.0003296518066431856",
            "extra": "mean: 12.300797000003172 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.754610322974447,
            "unit": "iter/sec",
            "range": "stddev: 0.01235191156389289",
            "extra": "mean: 43.94713800000076 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.52453329802051,
            "unit": "iter/sec",
            "range": "stddev: 0.00002620593230912173",
            "extra": "mean: 4.140366141463923 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.863865509350177,
            "unit": "iter/sec",
            "range": "stddev: 0.0007383440803365442",
            "extra": "mean: 33.485283399997456 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.51133275399558,
            "unit": "iter/sec",
            "range": "stddev: 0.0008788092802154955",
            "extra": "mean: 86.87091420000002 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 89.48608494883781,
            "unit": "iter/sec",
            "range": "stddev: 0.013904654006338902",
            "extra": "mean: 11.174921783333502 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3265.3574759057597,
            "unit": "iter/sec",
            "range": "stddev: 0.000009960866920327357",
            "extra": "mean: 306.24518368317865 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.718914384047515,
            "unit": "iter/sec",
            "range": "stddev: 0.0007136341837602341",
            "extra": "mean: 38.88189000000182 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.518093565995,
            "unit": "iter/sec",
            "range": "stddev: 0.013780626716454329",
            "extra": "mean: 51.23451205000009 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "29e2c3d168b4f2165e46bef284d6f4ccd318045a",
          "message": "docs: fix Returns rendering and tidy up docs styling (#36)\n\n* docs: fix Returns rendering in mkdocstrings\n\nIndent continuation lines in Returns docstring sections so griffe's\nGoogle-style parser treats them as a single entry instead of separate\nrows.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: change theme to deep orange and add bottom padding\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: tidy acknowledgements formatting and wording\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add `/commit` and `/pr` skills\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-09T19:16:44+01:00",
          "tree_id": "f4c0e87ff247b203a80df1624b4cb61340461453",
          "url": "https://github.com/adrhill/asdex/commit/29e2c3d168b4f2165e46bef284d6f4ccd318045a"
        },
        "date": 1770661039241,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 677.145367624834,
            "unit": "iter/sec",
            "range": "stddev: 0.003731948593928422",
            "extra": "mean: 1.476787773809361 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3181.6377425958362,
            "unit": "iter/sec",
            "range": "stddev: 0.000042526663793828266",
            "extra": "mean: 314.30353827275115 usec\nrounds: 2038"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.856557117330695,
            "unit": "iter/sec",
            "range": "stddev: 0.00030295826722973593",
            "extra": "mean: 17.284125600008338 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78.70789344148659,
            "unit": "iter/sec",
            "range": "stddev: 0.0003088845332743432",
            "extra": "mean: 12.70520599999827 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.258785715505567,
            "unit": "iter/sec",
            "range": "stddev: 0.013007492560140353",
            "extra": "mean: 44.926080550000336 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.93621187356382,
            "unit": "iter/sec",
            "range": "stddev: 0.000039593595009196616",
            "extra": "mean: 4.150476145631319 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.038709697686382,
            "unit": "iter/sec",
            "range": "stddev: 0.0004019125444288043",
            "extra": "mean: 34.436791800004585 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.348635814194116,
            "unit": "iter/sec",
            "range": "stddev: 0.0006530663978477299",
            "extra": "mean: 88.11631780000084 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 85.46581583166335,
            "unit": "iter/sec",
            "range": "stddev: 0.015358443790062229",
            "extra": "mean: 11.700584500002167 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3246.857676684084,
            "unit": "iter/sec",
            "range": "stddev: 0.000008115087955776888",
            "extra": "mean: 307.9900936776722 usec\nrounds: 2167"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.90928707773267,
            "unit": "iter/sec",
            "range": "stddev: 0.0004120102207146526",
            "extra": "mean: 38.596198999988474 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.17292231189587,
            "unit": "iter/sec",
            "range": "stddev: 0.015491420064798632",
            "extra": "mean: 52.15688999999486 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5ef98b5c3697eb80c4b316c45366b441e33d4d9d",
          "message": "feat: validate input shape against colored pattern (#37)\n\n* feat: validate input shape against colored pattern in jacobian/hessian\n\n- Add `input_shape` field to `SparsityPattern`\n- Set it during sparsity detection\n- Raise `ValueError` when the input shape doesn't match the precomputed pattern\n- Tests for the shape mismatch error and `ColoredPattern.__repr__`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add commit body to commit skill template\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T13:49:09+01:00",
          "tree_id": "d86c37176611f1dad5816d56b383798d10cae6fa",
          "url": "https://github.com/adrhill/asdex/commit/5ef98b5c3697eb80c4b316c45366b441e33d4d9d"
        },
        "date": 1770900583453,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 719.833868817538,
            "unit": "iter/sec",
            "range": "stddev: 0.00281169340054362",
            "extra": "mean: 1.3892094319522466 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3250.358198832145,
            "unit": "iter/sec",
            "range": "stddev: 0.000016023514380197455",
            "extra": "mean: 307.6583991140732 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.896325710578616,
            "unit": "iter/sec",
            "range": "stddev: 0.0005793893723020273",
            "extra": "mean: 16.42135199999899 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.65035078213562,
            "unit": "iter/sec",
            "range": "stddev: 0.00031867566911456316",
            "extra": "mean: 12.099162200000535 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.227954461359545,
            "unit": "iter/sec",
            "range": "stddev: 0.01110437429137803",
            "extra": "mean: 49.43653605263203 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.89627924202574,
            "unit": "iter/sec",
            "range": "stddev: 0.00016368438762870383",
            "extra": "mean: 4.116983607655777 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 23.847389261988344,
            "unit": "iter/sec",
            "range": "stddev: 0.01964484672127803",
            "extra": "mean: 41.93331140000112 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.548515658426918,
            "unit": "iter/sec",
            "range": "stddev: 0.0010206788107549611",
            "extra": "mean: 86.59121479999925 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.32167959160938,
            "unit": "iter/sec",
            "range": "stddev: 0.012334429811395198",
            "extra": "mean: 11.322248451613465 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3219.0430446066243,
            "unit": "iter/sec",
            "range": "stddev: 0.000005888728460267756",
            "extra": "mean: 310.65132902632644 usec\nrounds: 2167"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.7300705538602,
            "unit": "iter/sec",
            "range": "stddev: 0.0013814747982198487",
            "extra": "mean: 38.865031400000305 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.65256932154206,
            "unit": "iter/sec",
            "range": "stddev: 0.012069248409218416",
            "extra": "mean: 50.883931949999806 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "20c89579e4df0ab695892ab2b29f9b415391cc1c",
          "message": "build: replace pre-commit with prek",
          "timestamp": "2026-02-12T14:54:29+01:00",
          "tree_id": "2689ed4d19e5834e34d37251d6b920c08a14f37c",
          "url": "https://github.com/adrhill/asdex/commit/20c89579e4df0ab695892ab2b29f9b415391cc1c"
        },
        "date": 1770904504648,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 643.5313669814055,
            "unit": "iter/sec",
            "range": "stddev: 0.0044542216082207485",
            "extra": "mean: 1.5539258089169325 msec\nrounds: 157"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3224.23806200862,
            "unit": "iter/sec",
            "range": "stddev: 0.00002916052790188686",
            "extra": "mean: 310.15079555788907 usec\nrounds: 1981"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.15702813115722,
            "unit": "iter/sec",
            "range": "stddev: 0.0006391444736341843",
            "extra": "mean: 16.623161600000458 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 42.775986300461106,
            "unit": "iter/sec",
            "range": "stddev: 0.024326382584644674",
            "extra": "mean: 23.377602399999375 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.27935909236833,
            "unit": "iter/sec",
            "range": "stddev: 0.0018566868601466843",
            "extra": "mean: 41.187248649999475 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.69342933768402,
            "unit": "iter/sec",
            "range": "stddev: 0.0004251940551206898",
            "extra": "mean: 4.137472842105448 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.199918264526453,
            "unit": "iter/sec",
            "range": "stddev: 0.0007513129798931189",
            "extra": "mean: 33.112672399998644 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.540333424929234,
            "unit": "iter/sec",
            "range": "stddev: 0.002217146288181997",
            "extra": "mean: 86.65260899999794 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.60313812162175,
            "unit": "iter/sec",
            "range": "stddev: 0.00020022473470073033",
            "extra": "mean: 8.156398076923647 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3218.803335481656,
            "unit": "iter/sec",
            "range": "stddev: 0.000019914681849650234",
            "extra": "mean: 310.6744636979699 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.070076253194046,
            "unit": "iter/sec",
            "range": "stddev: 0.0014552440469540215",
            "extra": "mean: 39.88819140000004 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.838062094605796,
            "unit": "iter/sec",
            "range": "stddev: 0.01636441936288194",
            "extra": "mean: 53.08401655000097 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "08f32fa961090daebddf2529fcadd9004e545be0",
          "message": "feat(interpret): add precise handler for `top_k` primitive (#38)\n\nValues output uses reduction-along-last-axis sparsity (each output\ndepends on all inputs in its batch slice). Indices output has zero\nderivative. More precise than conservative fallback for batched inputs.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T15:17:31+01:00",
          "tree_id": "8651be7ee9313a8c141c10627ef89b2a0528945e",
          "url": "https://github.com/adrhill/asdex/commit/08f32fa961090daebddf2529fcadd9004e545be0"
        },
        "date": 1770905886118,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 722.9857522909932,
            "unit": "iter/sec",
            "range": "stddev: 0.003063088422791185",
            "extra": "mean: 1.3831531213875317 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3519.8497612231654,
            "unit": "iter/sec",
            "range": "stddev: 0.000005846734298674359",
            "extra": "mean: 284.1030350262719 usec\nrounds: 1713"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.30748009959075,
            "unit": "iter/sec",
            "range": "stddev: 0.0014145538334847418",
            "extra": "mean: 16.581690999998955 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 88.84159673010656,
            "unit": "iter/sec",
            "range": "stddev: 0.0002949782751300568",
            "extra": "mean: 11.255988600001388 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.055162644675807,
            "unit": "iter/sec",
            "range": "stddev: 0.009461796350496083",
            "extra": "mean: 41.571117800000934 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 259.2180117685354,
            "unit": "iter/sec",
            "range": "stddev: 0.0002773595661799193",
            "extra": "mean: 3.8577566164381123 msec\nrounds: 219"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.4929704320252,
            "unit": "iter/sec",
            "range": "stddev: 0.00047238008533564204",
            "extra": "mean: 31.753117799999586 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.351340179178306,
            "unit": "iter/sec",
            "range": "stddev: 0.021916167253562754",
            "extra": "mean: 88.09532479999973 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 101.8270910945465,
            "unit": "iter/sec",
            "range": "stddev: 0.009808564689977936",
            "extra": "mean: 9.820569253731303 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3485.344607331911,
            "unit": "iter/sec",
            "range": "stddev: 0.00000555427332546547",
            "extra": "mean: 286.9156748220419 usec\nrounds: 2248"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.104539397866976,
            "unit": "iter/sec",
            "range": "stddev: 0.00043993295837188205",
            "extra": "mean: 33.21758180000103 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 21.865084352220794,
            "unit": "iter/sec",
            "range": "stddev: 0.013924067876027369",
            "extra": "mean: 45.73501679166547 msec\nrounds: 24"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be0797e8b6a3296ae249f790351b4fe6ac6d2c32",
          "message": "feat(interpret): dispatch `scatter-mul`, `scatter-min`, `scatter-max` primitives (#39)\n\n* feat(interpret): dispatch `scatter-mul`, `scatter-min`, `scatter-max` primitives\n\nRoute the three remaining scatter combine variants through prop_scatter.\nThe existing logic already handles them correctly via the update_jaxpr\ncheck — only the dispatch case and variable naming needed updating.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(test): parametrize scatter combine tests\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T15:45:45+01:00",
          "tree_id": "203092706018877ec1f3b0abc99bd1e1a3a26f04",
          "url": "https://github.com/adrhill/asdex/commit/be0797e8b6a3296ae249f790351b4fe6ac6d2c32"
        },
        "date": 1770907579354,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 722.2947220769785,
            "unit": "iter/sec",
            "range": "stddev: 0.0026879170408861657",
            "extra": "mean: 1.384476404762411 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3231.9595786423333,
            "unit": "iter/sec",
            "range": "stddev: 0.000013487237915397988",
            "extra": "mean: 309.40981026132613 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.6559469977695,
            "unit": "iter/sec",
            "range": "stddev: 0.00046462734021231514",
            "extra": "mean: 16.486429599999042 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 76.44995870860156,
            "unit": "iter/sec",
            "range": "stddev: 0.0017894496803650183",
            "extra": "mean: 13.080451800001924 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.360653824333163,
            "unit": "iter/sec",
            "range": "stddev: 0.009192213273306632",
            "extra": "mean: 42.807021050000316 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.82145664731638,
            "unit": "iter/sec",
            "range": "stddev: 0.00004304719848695567",
            "extra": "mean: 4.152453913043565 msec\nrounds: 207"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.16700207867406,
            "unit": "iter/sec",
            "range": "stddev: 0.0008123277717946671",
            "extra": "mean: 32.08521619999658 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.560161708009309,
            "unit": "iter/sec",
            "range": "stddev: 0.023477852793254407",
            "extra": "mean: 94.69551960000331 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 98.22724488927103,
            "unit": "iter/sec",
            "range": "stddev: 0.009836096977747715",
            "extra": "mean: 10.180474888889265 msec\nrounds: 63"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3243.1624211003145,
            "unit": "iter/sec",
            "range": "stddev: 0.000006364695528610973",
            "extra": "mean: 308.3410172410446 usec\nrounds: 2146"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.80413596639022,
            "unit": "iter/sec",
            "range": "stddev: 0.0015930234754946624",
            "extra": "mean: 38.753477399998815 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.48316589060503,
            "unit": "iter/sec",
            "range": "stddev: 0.01135465483540263",
            "extra": "mean: 48.82057809523809 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a1102180c9bda9ca851452b9f86c895d234a15fc",
          "message": "feat(interpret): add precise handler for `platform_index` primitive (#40)\n\nEnables sparsity detection through `jax.lax.platform_dependent` and\nops that use it internally (e.g. `jnp.diag`). The output is a constant\nscalar with no input dependencies, so the handler is trivial.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T23:35:04+01:00",
          "tree_id": "4165087bfaa2a36eeec963228ebf1a769e75fca8",
          "url": "https://github.com/adrhill/asdex/commit/a1102180c9bda9ca851452b9f86c895d234a15fc"
        },
        "date": 1770935740847,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 682.6385393583553,
            "unit": "iter/sec",
            "range": "stddev: 0.0041925798684314805",
            "extra": "mean: 1.4649041071427757 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3495.11698356521,
            "unit": "iter/sec",
            "range": "stddev: 0.000007538006808316196",
            "extra": "mean: 286.11345620252905 usec\nrounds: 1975"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.814293908406455,
            "unit": "iter/sec",
            "range": "stddev: 0.00024124658769941485",
            "extra": "mean: 16.44350259999925 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.480956883445,
            "unit": "iter/sec",
            "range": "stddev: 0.00025827983433757726",
            "extra": "mean: 11.978779799999018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.697651990430153,
            "unit": "iter/sec",
            "range": "stddev: 0.012758314959740056",
            "extra": "mean: 42.19827350000038 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.10697416341662,
            "unit": "iter/sec",
            "range": "stddev: 0.0006264984177472939",
            "extra": "mean: 3.998289145454269 msec\nrounds: 220"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.2947281312716,
            "unit": "iter/sec",
            "range": "stddev: 0.00034950355629731337",
            "extra": "mean: 34.1358348 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.160176771795076,
            "unit": "iter/sec",
            "range": "stddev: 0.028674849207392075",
            "extra": "mean: 98.42348439999853 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.44099091402681,
            "unit": "iter/sec",
            "range": "stddev: 0.013586433887862445",
            "extra": "mean: 11.306974171875762 msec\nrounds: 64"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3439.209209553748,
            "unit": "iter/sec",
            "range": "stddev: 0.00002398748706437811",
            "extra": "mean: 290.7645156398479 usec\nrounds: 2110"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.88348071995565,
            "unit": "iter/sec",
            "range": "stddev: 0.00024861250358308224",
            "extra": "mean: 37.19756419999953 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.95599433344662,
            "unit": "iter/sec",
            "range": "stddev: 0.020020676950717763",
            "extra": "mean: 52.75376128571452 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "75f0b6e7343a883b7250de7fb14cb8a04c55cf65",
          "message": "feat(interpret): add precise handler for `scan` primitive (#41)\n\n* feat(interpret): add precise handler for `scan` primitive\n\nPropagate index sets through scan bodies using fixed-point iteration\non the carry, same strategy as while_loop. Also extract shared\n_MAX_FIXED_POINT_ITERS constant and raise RuntimeError on\nnon-convergence in both scan and while_loop.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): add edge-case and pytree test coverage for scan\n\nAdd 10 new tests for scan propagation: pytree xs/ys, length=1,\nscalar carry, unroll, ys-independent-of-carry, carry mixing,\ncarry tuple interaction, scan+cond composition, and pytree\nJacobian values.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): add tests for `associative_scan` decomposition\n\n`associative_scan` is not a JAX primitive — it decomposes into\nslice/add/pad/concatenate/rev, all of which have precise handlers.\nRemove the unreachable `prop_throw_error` dispatch entry and add\n8 tests verifying correct sparsity patterns through decomposition.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T00:35:17+01:00",
          "tree_id": "1ca5aeeb97d7b0389a38833c4af7bd42194232e0",
          "url": "https://github.com/adrhill/asdex/commit/75f0b6e7343a883b7250de7fb14cb8a04c55cf65"
        },
        "date": 1770939349410,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 677.5719615299944,
            "unit": "iter/sec",
            "range": "stddev: 0.003223352527064714",
            "extra": "mean: 1.475858000000392 msec\nrounds: 162"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3193.7044137508724,
            "unit": "iter/sec",
            "range": "stddev: 0.000040436034306314846",
            "extra": "mean: 313.11601527504604 usec\nrounds: 1964"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.32404096174191,
            "unit": "iter/sec",
            "range": "stddev: 0.0004958608321060904",
            "extra": "mean: 16.577138800005287 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.41699572296407,
            "unit": "iter/sec",
            "range": "stddev: 0.0004975323038944215",
            "extra": "mean: 12.282447799998408 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.980322295872117,
            "unit": "iter/sec",
            "range": "stddev: 0.010456052468856501",
            "extra": "mean: 43.51549064999958 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.7468017794468,
            "unit": "iter/sec",
            "range": "stddev: 0.000033911002445356324",
            "extra": "mean: 4.136559377990578 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.22487811047109,
            "unit": "iter/sec",
            "range": "stddev: 0.0008562361882805677",
            "extra": "mean: 33.08532780000064 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.12057296409335,
            "unit": "iter/sec",
            "range": "stddev: 0.024852406334139383",
            "extra": "mean: 98.80863500000316 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.54643962133174,
            "unit": "iter/sec",
            "range": "stddev: 0.011824979326540029",
            "extra": "mean: 10.805385967214477 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3241.553047861711,
            "unit": "iter/sec",
            "range": "stddev: 0.000011370589565621818",
            "extra": "mean: 308.494103053365 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.324174070120854,
            "unit": "iter/sec",
            "range": "stddev: 0.000377434627514299",
            "extra": "mean: 42.87397260000034 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.529680287023563,
            "unit": "iter/sec",
            "range": "stddev: 0.013524877257882071",
            "extra": "mean: 53.967471888886585 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "14d521b2e0ec2a6d281547a44397ea1fe323f10c",
          "message": "test: add further Jacobian and Hessian sparsity tests (#42)\n\n* test: add SCT Global Jacobian and Hessian test cases\n\nPort 26 test cases from SparseConnectivityTracer.jl's Global Jacobian\nand Global Hessian testsets covering element-wise ops, conditional\nbranching, clamp interactions, composite functions, and various\nHessian sparsity patterns.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: add scalar and binary sparsity tests\n\nAdd test_scalar.py with parametrized Jacobian/Hessian tests for unary\n(R->R) and binary (R^2->R) functions covering nonlinear, linear,\nzero-derivative, and constant categories. Extend test_sct.py with\ncomposite function Jacobians and AMPGO07 benchmark. Add TODO for\nauto-squeezing (1,)-shaped output in hessian_sparsity.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(test): distribute test_sct.py into specific test files\n\nMove tests from test_sct.py to their natural homes: multi-variable\npatterns to test_detection.py, where/select Hessians to test_select.py,\nand scalar multiply-by-zero to test_scalar.py. Drop two duplicate\nJacobian where tests already in test_select.py. Remove SCT references\nfrom test docstrings.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T01:04:37+01:00",
          "tree_id": "c7b97f30a43100132fb6f21993c7f176266395d3",
          "url": "https://github.com/adrhill/asdex/commit/14d521b2e0ec2a6d281547a44397ea1fe323f10c"
        },
        "date": 1770941111644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 727.4552136878331,
            "unit": "iter/sec",
            "range": "stddev: 0.0027085314256770636",
            "extra": "mean: 1.374655073170074 msec\nrounds: 164"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3250.6322520591775,
            "unit": "iter/sec",
            "range": "stddev: 0.000010122877984471044",
            "extra": "mean: 307.6324611516821 usec\nrounds: 2188"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.44268883429403,
            "unit": "iter/sec",
            "range": "stddev: 0.0003676199250515747",
            "extra": "mean: 16.275329400002647 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.83707985843277,
            "unit": "iter/sec",
            "range": "stddev: 0.0005073497606632555",
            "extra": "mean: 12.071888599996328 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.474296959974954,
            "unit": "iter/sec",
            "range": "stddev: 0.008539089758487146",
            "extra": "mean: 42.599784850002465 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.0727813514597,
            "unit": "iter/sec",
            "range": "stddev: 0.000028169016294428586",
            "extra": "mean: 4.148124870812775 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.09607578531529,
            "unit": "iter/sec",
            "range": "stddev: 0.0005306354178229245",
            "extra": "mean: 32.15839860000074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.6617692833323,
            "unit": "iter/sec",
            "range": "stddev: 0.0009188242088559065",
            "extra": "mean: 85.75028160000215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 127.01147006716637,
            "unit": "iter/sec",
            "range": "stddev: 0.00018014594367225624",
            "extra": "mean: 7.873304666666551 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3273.1664819843086,
            "unit": "iter/sec",
            "range": "stddev: 0.000010519743356701659",
            "extra": "mean: 305.51455463816336 usec\nrounds: 2059"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.152303070632144,
            "unit": "iter/sec",
            "range": "stddev: 0.0004760465592249132",
            "extra": "mean: 41.40391900000395 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.92967385755103,
            "unit": "iter/sec",
            "range": "stddev: 0.013274601161672946",
            "extra": "mean: 52.82711194736728 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d0cf6c855daafb2c2c89bfee477285c0970cdf63",
          "message": "feat: auto-squeeze scalar functions in Hessian API (#43)\n\nFunctions returning shape (1,) or (1, 1) are now automatically squeezed\nto scalar in hessian_sparsity, hessian, and _eval_hessian. Non-scalar\noutputs like (3,) raise ValueError with a clear message.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T01:20:15+01:00",
          "tree_id": "f49c2009bdf44805368feebffef7c95d5eb0fba9",
          "url": "https://github.com/adrhill/asdex/commit/d0cf6c855daafb2c2c89bfee477285c0970cdf63"
        },
        "date": 1770942048460,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 721.1351423090557,
            "unit": "iter/sec",
            "range": "stddev: 0.0026441433518342708",
            "extra": "mean: 1.3867026321835132 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3144.214328351857,
            "unit": "iter/sec",
            "range": "stddev: 0.000049835068120109474",
            "extra": "mean: 318.0444764794971 usec\nrounds: 1977"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.78561929148982,
            "unit": "iter/sec",
            "range": "stddev: 0.0005269357298390231",
            "extra": "mean: 16.72643039999997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.96478020723963,
            "unit": "iter/sec",
            "range": "stddev: 0.00027498864592990667",
            "extra": "mean: 12.351049400002978 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.76357009312604,
            "unit": "iter/sec",
            "range": "stddev: 0.00039189592060926223",
            "extra": "mean: 40.38189955 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.4630597217845,
            "unit": "iter/sec",
            "range": "stddev: 0.00023481792141646113",
            "extra": "mean: 4.1935216346158715 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.180190707901723,
            "unit": "iter/sec",
            "range": "stddev: 0.0009931302215204461",
            "extra": "mean: 33.13431679999894 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.840995746276514,
            "unit": "iter/sec",
            "range": "stddev: 0.0016599872266401534",
            "extra": "mean: 84.45235699999785 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.97123639950034,
            "unit": "iter/sec",
            "range": "stddev: 0.009764910214135876",
            "extra": "mean: 10.756014857143217 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3260.1031224121384,
            "unit": "iter/sec",
            "range": "stddev: 0.000009933553293226288",
            "extra": "mean: 306.73876330025524 usec\nrounds: 2218"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.746880737347322,
            "unit": "iter/sec",
            "range": "stddev: 0.00024044901602949465",
            "extra": "mean: 40.409133200000724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.887063444266953,
            "unit": "iter/sec",
            "range": "stddev: 0.012850195064945185",
            "extra": "mean: 52.946293263156456 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "009c32c2da6487adaf1bee356f91323a29c1944f",
          "message": "feat(interpret): add precise handlers for `sort`, `split`, and `tile` (#44)\n\n* feat(interpret): add precise handlers for `split` and `tile` primitives\n\n- Add `_split.py`: partitions input along an axis, each output element\n  maps to exactly one input element (selection matrix Jacobian)\n- Add `_tile.py`: repeats input via modular indexing, each output element\n  depends on exactly one input element\n- Convert `prop_reshape` size-mismatch branch from conservative fallback\n  to ValueError (should never occur in valid JAX code)\n- Remove `split` and `tile` from conservative fallback group\n- Delete TODO.md (all items resolved)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* chore(test): suppress pytest progress output for coding agents\n\nAdd --tb=short -q --no-header defaults and a conftest hook\nto only show output on failures, reducing noise for AI agents.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat(interpret): add precise handler for `sort` primitive\n\nSort along one dimension only mixes elements within slices along that\ndimension, producing block-diagonal patterns for multi-dimensional arrays.\nThe conservative fallback was correct for 1D but overly dense for nD.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add TODO for conservative fallback primitives\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): simplify sort handler with moveaxis grouping\n\nReplace np.indices/ravel_multi_index batch mapping and 1D special case\nwith a single moveaxis+reshape to group flat indices by batch coordinates.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): use union_all to shorten sort handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T16:16:03+01:00",
          "tree_id": "c1e02f55448ce6f0caa17a9c359348d8ed74ceb9",
          "url": "https://github.com/adrhill/asdex/commit/009c32c2da6487adaf1bee356f91323a29c1944f"
        },
        "date": 1770995798173,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 690.9142078096987,
            "unit": "iter/sec",
            "range": "stddev: 0.0033936562156871588",
            "extra": "mean: 1.4473577018630857 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3191.0828839391606,
            "unit": "iter/sec",
            "range": "stddev: 0.000009173216417859197",
            "extra": "mean: 313.3732455001521 usec\nrounds: 2000"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.27249555391687,
            "unit": "iter/sec",
            "range": "stddev: 0.00047878416547247963",
            "extra": "mean: 16.871231599998282 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.47854215120998,
            "unit": "iter/sec",
            "range": "stddev: 0.0005675141620069814",
            "extra": "mean: 12.42567239999346 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.30256428619717,
            "unit": "iter/sec",
            "range": "stddev: 0.016506883476296964",
            "extra": "mean: 46.9427054210531 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.9763874247778,
            "unit": "iter/sec",
            "range": "stddev: 0.00004580951208033439",
            "extra": "mean: 4.184513837438304 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.95862863306994,
            "unit": "iter/sec",
            "range": "stddev: 0.00047804367689638296",
            "extra": "mean: 35.76713339999742 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.139021488914315,
            "unit": "iter/sec",
            "range": "stddev: 0.0007201386855225613",
            "extra": "mean: 89.77449240000226 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 79.730616940342,
            "unit": "iter/sec",
            "range": "stddev: 0.01620858435274737",
            "extra": "mean: 12.542233314816121 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3112.5766852491256,
            "unit": "iter/sec",
            "range": "stddev: 0.00000803711231821955",
            "extra": "mean: 321.2772249882613 usec\nrounds: 2089"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.024054251745774,
            "unit": "iter/sec",
            "range": "stddev: 0.0004455812829041531",
            "extra": "mean: 43.432837199998175 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.100853164129088,
            "unit": "iter/sec",
            "range": "stddev: 0.01869145448156639",
            "extra": "mean: 58.47661461111247 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "306e7e9c88dc4c2f230f56cbf95c11717b10b281",
          "message": "refactor(interpret): extract common patterns into _commons.py (#45)\n\n* refactor(interpret): extract common patterns into _commons.py\n\nAdd `permute_indices`, `position_map`, `fixed_point_loop`, and\n`conservative_indices` utilities to reduce duplication across handlers.\n\n- permute_indices: replaces repeated [in_indices[j].copy() for j in map]\n  pattern across 8+ handler files\n- position_map: replaces repeated np.arange(n).reshape(shape) pattern\n- fixed_point_loop: extracts ~22-line fixed-point iteration from\n  _while.py and _scan.py\n- Rename conservative_deps → conservative_indices for consistency\n- Simplify _transpose.py and _slice.py using numpy iota approach\n- Update CLAUDE.md with new naming conventions\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): standardize variable names across handlers\n\nUse `permutation_map` consistently for the flat map passed to\n`permute_indices` (was `perm`, `chunk_indices` in some files).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): reorganize and improve CLAUDE.md\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Remove skills\n\n* refactor(interpret): rename *_deps variables to *_indices\n\nRename list[IndexSets] variables from *_deps to *_indices across\nhandlers, and standardize dim/const_inputs naming for consistency.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* style(interpret): reorganize _commons.py by concept\n\nReorder functions into logical groups and replace banner comments\nwith lightweight section headers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T17:10:38+01:00",
          "tree_id": "2d1e15584dc444162aa78f05a5a3112aeac9c8b3",
          "url": "https://github.com/adrhill/asdex/commit/306e7e9c88dc4c2f230f56cbf95c11717b10b281"
        },
        "date": 1771344672881,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 887.7797741654978,
            "unit": "iter/sec",
            "range": "stddev: 0.0028226101101665956",
            "extra": "mean: 1.1264054770114444 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3238.7044076284596,
            "unit": "iter/sec",
            "range": "stddev: 0.00000939234181852142",
            "extra": "mean: 308.7654426395306 usec\nrounds: 1970"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.66982201512308,
            "unit": "iter/sec",
            "range": "stddev: 0.0005524263511376948",
            "extra": "mean: 16.482659200001137 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.75666471742497,
            "unit": "iter/sec",
            "range": "stddev: 0.0003307333416506016",
            "extra": "mean: 11.798482200001104 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.97196157982859,
            "unit": "iter/sec",
            "range": "stddev: 0.010746329791815787",
            "extra": "mean: 45.51254999999873 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.70633722754758,
            "unit": "iter/sec",
            "range": "stddev: 0.00035484520475322166",
            "extra": "mean: 4.137251887850083 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.921926285850002,
            "unit": "iter/sec",
            "range": "stddev: 0.0007166286924814365",
            "extra": "mean: 32.339511800000764 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.173761870577348,
            "unit": "iter/sec",
            "range": "stddev: 0.027639930393593512",
            "extra": "mean: 98.29205880000131 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 91.49734793493127,
            "unit": "iter/sec",
            "range": "stddev: 0.011482099347955155",
            "extra": "mean: 10.929278526314821 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3322.205091691788,
            "unit": "iter/sec",
            "range": "stddev: 0.000010898572352751898",
            "extra": "mean: 301.0048965672868 usec\nrounds: 2214"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.745150927852492,
            "unit": "iter/sec",
            "range": "stddev: 0.0003438746288250086",
            "extra": "mean: 40.41195799999855 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.225298006730156,
            "unit": "iter/sec",
            "range": "stddev: 0.012048987092800638",
            "extra": "mean: 52.01479840000047 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c058eb137c4fd7347790f11b05049ef861d4d470",
          "message": "feat(interpret): validate that auxiliary inputs have no index sets (#46)\n\n* feat(interpret): validate that auxiliary inputs have no index sets\n\nAdd `check_no_index_sets` to `_commons.py` and call it in handlers\nthat silently ignore auxiliary input dependencies (gather, scatter,\ndynamic_slice, dynamic_update_slice, conv).\nThis turns an incorrect-result scenario into an explicit error\nwith a link to the issue tracker.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): improve error messages with contribution prompt\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T17:38:19+01:00",
          "tree_id": "742ef1de5e5e12919e3cb83196fca2e264ff1fc0",
          "url": "https://github.com/adrhill/asdex/commit/c058eb137c4fd7347790f11b05049ef861d4d470"
        },
        "date": 1771346335230,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 907.2011507143188,
            "unit": "iter/sec",
            "range": "stddev: 0.002562102892464047",
            "extra": "mean: 1.1022913707865256 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3309.835067150202,
            "unit": "iter/sec",
            "range": "stddev: 0.00002930650476216587",
            "extra": "mean: 302.1298583500141 usec\nrounds: 2012"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.05211226218607,
            "unit": "iter/sec",
            "range": "stddev: 0.0004533062844772982",
            "extra": "mean: 16.652203600000348 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.89537783282647,
            "unit": "iter/sec",
            "range": "stddev: 0.0002874840349272712",
            "extra": "mean: 11.919607799998744 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.977162313798104,
            "unit": "iter/sec",
            "range": "stddev: 0.012082352870536843",
            "extra": "mean: 45.50177978947545 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.6088576956201,
            "unit": "iter/sec",
            "range": "stddev: 0.000033380100138361045",
            "extra": "mean: 4.006268083720921 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.67801813057131,
            "unit": "iter/sec",
            "range": "stddev: 0.0008903631565513651",
            "extra": "mean: 32.596629800002574 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.551705826883836,
            "unit": "iter/sec",
            "range": "stddev: 0.0018878810816523204",
            "extra": "mean: 86.56730140000093 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 125.52423813601064,
            "unit": "iter/sec",
            "range": "stddev: 0.00012179748303217716",
            "extra": "mean: 7.966588882351623 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3294.259441933651,
            "unit": "iter/sec",
            "range": "stddev: 0.000009281205300355939",
            "extra": "mean: 303.55836194037715 usec\nrounds: 2144"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.086868037055886,
            "unit": "iter/sec",
            "range": "stddev: 0.0004054246260009299",
            "extra": "mean: 41.51639800000453 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.848864788837783,
            "unit": "iter/sec",
            "range": "stddev: 0.0001705477214420844",
            "extra": "mean: 50.38071500000143 msec\nrounds: 10"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e2c74a45c0a476d3264e2fcc91a299666504b897",
          "message": "style: replace decorative block comments with plain comments (#47)\n\n* style: replace decorative block comments with plain comments\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T17:46:42+01:00",
          "tree_id": "b8dd20dd152e00867cf74ff9d11418ba1624477d",
          "url": "https://github.com/adrhill/asdex/commit/e2c74a45c0a476d3264e2fcc91a299666504b897"
        },
        "date": 1771346835811,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 895.3563660514942,
            "unit": "iter/sec",
            "range": "stddev: 0.0029843897186507956",
            "extra": "mean: 1.1168737252743086 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3349.833643737129,
            "unit": "iter/sec",
            "range": "stddev: 0.000008166385634820083",
            "extra": "mean: 298.5222868812028 usec\nrounds: 2081"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.93402941515513,
            "unit": "iter/sec",
            "range": "stddev: 0.0006845230057252801",
            "extra": "mean: 16.68501199999639 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.56250715929335,
            "unit": "iter/sec",
            "range": "stddev: 0.0002850108147459759",
            "extra": "mean: 11.825571799997192 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.47406642659974,
            "unit": "iter/sec",
            "range": "stddev: 0.010802625071786",
            "extra": "mean: 44.495730368422564 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.38295598678462,
            "unit": "iter/sec",
            "range": "stddev: 0.000032724505154932973",
            "extra": "mean: 4.009897132075828 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 20.555906282712566,
            "unit": "iter/sec",
            "range": "stddev: 0.03016463244369432",
            "extra": "mean: 48.64781859999994 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.416835730404872,
            "unit": "iter/sec",
            "range": "stddev: 0.0014671698933717938",
            "extra": "mean: 87.58994380000047 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 82.3544950076212,
            "unit": "iter/sec",
            "range": "stddev: 0.016120217489593452",
            "extra": "mean: 12.142628036362298 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3363.0489590851657,
            "unit": "iter/sec",
            "range": "stddev: 0.000014709834907808221",
            "extra": "mean: 297.3492245179878 usec\nrounds: 1452"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.357447609176244,
            "unit": "iter/sec",
            "range": "stddev: 0.0007014304993980885",
            "extra": "mean: 41.055204799999956 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.052117964141,
            "unit": "iter/sec",
            "range": "stddev: 0.012456449977708485",
            "extra": "mean: 52.48760278947217 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "cc94199dc750725dee5cc07675d086bf16f39be7",
          "message": "docs: shorten acknowledgements",
          "timestamp": "2026-02-17T17:50:08+01:00",
          "tree_id": "8bd7686b3c7e4f8e423f153e559618cb66c5f1e8",
          "url": "https://github.com/adrhill/asdex/commit/cc94199dc750725dee5cc07675d086bf16f39be7"
        },
        "date": 1771347046708,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 871.0384654514165,
            "unit": "iter/sec",
            "range": "stddev: 0.003301625432639591",
            "extra": "mean: 1.1480549248554126 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3279.9187003593042,
            "unit": "iter/sec",
            "range": "stddev: 0.00003263873323820441",
            "extra": "mean: 304.8856058201849 usec\nrounds: 1890"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.44568099653483,
            "unit": "iter/sec",
            "range": "stddev: 0.0004458521104396105",
            "extra": "mean: 16.5437791999949 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.96408296329852,
            "unit": "iter/sec",
            "range": "stddev: 0.0004478092277082204",
            "extra": "mean: 11.76967919999754 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.510432531042596,
            "unit": "iter/sec",
            "range": "stddev: 0.010672449601116109",
            "extra": "mean: 44.42384652631478 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.49786616382855,
            "unit": "iter/sec",
            "range": "stddev: 0.00003737099667877332",
            "extra": "mean: 4.07335516037711 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.916384005384813,
            "unit": "iter/sec",
            "range": "stddev: 0.0005326903285155367",
            "extra": "mean: 32.345309199996564 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.164103826848775,
            "unit": "iter/sec",
            "range": "stddev: 0.025810664752011138",
            "extra": "mean: 98.38545700000338 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.14650392203067,
            "unit": "iter/sec",
            "range": "stddev: 0.012938536998624727",
            "extra": "mean: 11.344749428571125 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3378.3095616257697,
            "unit": "iter/sec",
            "range": "stddev: 0.00000987223414928293",
            "extra": "mean: 296.0060295714175 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.366165970386394,
            "unit": "iter/sec",
            "range": "stddev: 0.0006311067479173439",
            "extra": "mean: 41.04051499999457 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.668754281445153,
            "unit": "iter/sec",
            "range": "stddev: 0.015304957419392397",
            "extra": "mean: 53.56543800000081 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "eb5fa4de7d9df88a443e8a5cb4e2926c0662e97f",
          "message": "docs: shorten acknowledgements",
          "timestamp": "2026-02-17T17:51:11+01:00",
          "tree_id": "5d7f6a5da9c8e7cbb71e28d0f5a84ea8c7b0b3b6",
          "url": "https://github.com/adrhill/asdex/commit/eb5fa4de7d9df88a443e8a5cb4e2926c0662e97f"
        },
        "date": 1771347114292,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 824.7115165517603,
            "unit": "iter/sec",
            "range": "stddev: 0.003519076207313741",
            "extra": "mean: 1.2125452111802035 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3270.479449794303,
            "unit": "iter/sec",
            "range": "stddev: 0.0000365374570701765",
            "extra": "mean: 305.7655659823501 usec\nrounds: 2046"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.852647640304504,
            "unit": "iter/sec",
            "range": "stddev: 0.0005640149218572437",
            "extra": "mean: 17.285293599999818 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78.7457143235573,
            "unit": "iter/sec",
            "range": "stddev: 0.0005713139425574561",
            "extra": "mean: 12.699103799999989 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.363431704423686,
            "unit": "iter/sec",
            "range": "stddev: 0.013817636663093245",
            "extra": "mean: 44.715856368420916 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.39634721134718,
            "unit": "iter/sec",
            "range": "stddev: 0.00003402390819301327",
            "extra": "mean: 4.0258240961536895 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.51044814410414,
            "unit": "iter/sec",
            "range": "stddev: 0.000540736487535653",
            "extra": "mean: 35.07486079999751 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.547173143355772,
            "unit": "iter/sec",
            "range": "stddev: 0.001232261856978557",
            "extra": "mean: 86.60128219999876 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 124.27185348503144,
            "unit": "iter/sec",
            "range": "stddev: 0.0001571928306139954",
            "extra": "mean: 8.046874428572437 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3369.0482797394984,
            "unit": "iter/sec",
            "range": "stddev: 0.000009526468166210634",
            "extra": "mean: 296.81972977761006 usec\nrounds: 2250"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.401066769454253,
            "unit": "iter/sec",
            "range": "stddev: 0.00038090215105612753",
            "extra": "mean: 42.73309459999979 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.14602363080204,
            "unit": "iter/sec",
            "range": "stddev: 0.014614534069306845",
            "extra": "mean: 55.108492105264645 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eef8ea44791c20d5d98e1a2dbde6b443e0c7bc87",
          "message": "feat(interpret): add precise handler for `select_if_vmap` (#48)\n\n* feat(interpret): add precise handler for `select_if_vmap`\n\nReplace conservative fallback with element-wise union of both branches,\nmatching the existing `select_n` handler. This Equinox primitive appears\nwhen vmapping `lax.cond`.\n\nEquinox-specific handlers live in `_interpret/_equinox/`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Delete TODO.md\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T18:17:49+01:00",
          "tree_id": "25eaa57189f044bee91577f16ac81633e07cd84a",
          "url": "https://github.com/adrhill/asdex/commit/eef8ea44791c20d5d98e1a2dbde6b443e0c7bc87"
        },
        "date": 1771348707903,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 794.7581487868484,
            "unit": "iter/sec",
            "range": "stddev: 0.004325144315467476",
            "extra": "mean: 1.2582444125001313 msec\nrounds: 160"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3299.206909097776,
            "unit": "iter/sec",
            "range": "stddev: 0.000008122032421213656",
            "extra": "mean: 303.10314798457637 usec\nrounds: 1811"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.99341974493999,
            "unit": "iter/sec",
            "range": "stddev: 0.0012365262502901367",
            "extra": "mean: 17.545885199997713 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 38.870789569521726,
            "unit": "iter/sec",
            "range": "stddev: 0.029183120711244577",
            "extra": "mean: 25.72625899999963 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.49567861794613,
            "unit": "iter/sec",
            "range": "stddev: 0.015405448979915474",
            "extra": "mean: 46.52097836842092 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.74740153217584,
            "unit": "iter/sec",
            "range": "stddev: 0.00008691777184810932",
            "extra": "mean: 4.036369276995733 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.127138760652695,
            "unit": "iter/sec",
            "range": "stddev: 0.0013482713480052933",
            "extra": "mean: 34.332242799999335 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.752658668866257,
            "unit": "iter/sec",
            "range": "stddev: 0.03038615825784973",
            "extra": "mean: 102.53614260000035 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 83.60090009188005,
            "unit": "iter/sec",
            "range": "stddev: 0.015485337813165832",
            "extra": "mean: 11.961593701754026 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3300.055113460094,
            "unit": "iter/sec",
            "range": "stddev: 0.000010027350713909727",
            "extra": "mean: 303.0252421910324 usec\nrounds: 2209"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.482686081700628,
            "unit": "iter/sec",
            "range": "stddev: 0.0003865794368745746",
            "extra": "mean: 42.58456620000004 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.514549784099458,
            "unit": "iter/sec",
            "range": "stddev: 0.016385578812295583",
            "extra": "mean: 57.095387111111904 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "13fabd8e79dba77f2a2dd96261d06e920ee3e2f2",
          "message": "test: add Flax tests for ResNet and ViT (#49)\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T18:50:31+01:00",
          "tree_id": "9cf50e525acbf3d08e1bc76d246602ef308b32e4",
          "url": "https://github.com/adrhill/asdex/commit/13fabd8e79dba77f2a2dd96261d06e920ee3e2f2"
        },
        "date": 1771350669579,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 848.1519242675578,
            "unit": "iter/sec",
            "range": "stddev: 0.0034803017740738353",
            "extra": "mean: 1.1790340520226659 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3302.1729983386026,
            "unit": "iter/sec",
            "range": "stddev: 0.000009883536333398007",
            "extra": "mean: 302.8308936276575 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.58449349304892,
            "unit": "iter/sec",
            "range": "stddev: 0.00039309783387273363",
            "extra": "mean: 16.782890000007455 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.4372846881196,
            "unit": "iter/sec",
            "range": "stddev: 0.0003901038652539448",
            "extra": "mean: 11.84310939999591 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.58239515505839,
            "unit": "iter/sec",
            "range": "stddev: 0.010303401372151659",
            "extra": "mean: 44.28228242104793 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 244.4707645469658,
            "unit": "iter/sec",
            "range": "stddev: 0.000036800575128853796",
            "extra": "mean: 4.090468657277373 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.148011239108893,
            "unit": "iter/sec",
            "range": "stddev: 0.00108083669409251",
            "extra": "mean: 33.169683799997074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.988891393836056,
            "unit": "iter/sec",
            "range": "stddev: 0.02812034583355739",
            "extra": "mean: 100.11120960000426 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.90259443405586,
            "unit": "iter/sec",
            "range": "stddev: 0.01152334325275721",
            "extra": "mean: 10.427246581818137 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3331.3538941601664,
            "unit": "iter/sec",
            "range": "stddev: 0.00000881146345170166",
            "extra": "mean: 300.17825537928917 usec\nrounds: 2138"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.153194904191977,
            "unit": "iter/sec",
            "range": "stddev: 0.0007530479283163494",
            "extra": "mean: 41.40239019999967 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.747680535731813,
            "unit": "iter/sec",
            "range": "stddev: 0.012172765228445763",
            "extra": "mean: 53.33993173683899 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "325e56cf39767a21ab5ffe52c20160d1d2c7c978",
          "message": "docs: update README",
          "timestamp": "2026-02-18T16:09:31+01:00",
          "tree_id": "1573b6be4af37e8cf42236cc3aeb4348467d9a93",
          "url": "https://github.com/adrhill/asdex/commit/325e56cf39767a21ab5ffe52c20160d1d2c7c978"
        },
        "date": 1771427406553,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 887.6147949517167,
            "unit": "iter/sec",
            "range": "stddev: 0.003667573302644708",
            "extra": "mean: 1.126614839779002 msec\nrounds: 181"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3546.3273002372307,
            "unit": "iter/sec",
            "range": "stddev: 0.00000865034274968596",
            "extra": "mean: 281.98186894173733 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 66.5987075694935,
            "unit": "iter/sec",
            "range": "stddev: 0.000590092441710684",
            "extra": "mean: 15.015306400000838 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 94.04145621991354,
            "unit": "iter/sec",
            "range": "stddev: 0.00029505833654987646",
            "extra": "mean: 10.633608199999856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.275851685500108,
            "unit": "iter/sec",
            "range": "stddev: 0.011863085479565313",
            "extra": "mean: 47.00164368421118 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.5534595176675,
            "unit": "iter/sec",
            "range": "stddev: 0.00029745719547915423",
            "extra": "mean: 4.105874751195885 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 32.176384038682585,
            "unit": "iter/sec",
            "range": "stddev: 0.0006900718021585829",
            "extra": "mean: 31.07869420000071 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.683695699884238,
            "unit": "iter/sec",
            "range": "stddev: 0.0009647985070291153",
            "extra": "mean: 85.58935679999848 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 131.55487469048475,
            "unit": "iter/sec",
            "range": "stddev: 0.0002750710137421722",
            "extra": "mean: 7.60139069230803 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3526.538252676647,
            "unit": "iter/sec",
            "range": "stddev: 0.000009835950106978108",
            "extra": "mean: 283.56420045663725 usec\nrounds: 2190"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.64695246958439,
            "unit": "iter/sec",
            "range": "stddev: 0.00018536219295699606",
            "extra": "mean: 37.527743600001884 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.130791477727158,
            "unit": "iter/sec",
            "range": "stddev: 0.014416998759423516",
            "extra": "mean: 49.675145714285826 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca7ac725914be33aadd47b1a78c2de9933833b39",
          "message": "feat(verify): add correctness check utilities (#50)\n\nVerify sparse Jacobians/Hessians against JAX's dense references\nto catch bugs where sparsity detection misses nonzeros.\n\n- Add src/asdex/verify.py with VerificationError exception\n- Export new functions and exception from __init__.py\n- Add tests for success and failure cases\n- Add verification sections to Jacobian/Hessian how-to guides\n- Add autodoc entries to reference pages\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-18T17:08:37+01:00",
          "tree_id": "d06e33916ebfcd6bbbd93937747d54217e16dd25",
          "url": "https://github.com/adrhill/asdex/commit/ca7ac725914be33aadd47b1a78c2de9933833b39"
        },
        "date": 1771430955334,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 918.8605859459819,
            "unit": "iter/sec",
            "range": "stddev: 0.002437972477932767",
            "extra": "mean: 1.0883043796796266 msec\nrounds: 187"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3332.409597918421,
            "unit": "iter/sec",
            "range": "stddev: 0.000009595319594334697",
            "extra": "mean: 300.08315923248057 usec\nrounds: 2085"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.3353690808572,
            "unit": "iter/sec",
            "range": "stddev: 0.000434996176379028",
            "extra": "mean: 16.57402640000214 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.67022311971093,
            "unit": "iter/sec",
            "range": "stddev: 0.00023865480859403327",
            "extra": "mean: 11.951683199998797 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.954167201720836,
            "unit": "iter/sec",
            "range": "stddev: 0.009227671391487405",
            "extra": "mean: 43.56507431578836 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.5233278954595,
            "unit": "iter/sec",
            "range": "stddev: 0.000036277982511482205",
            "extra": "mean: 4.02376713875587 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.909148017958167,
            "unit": "iter/sec",
            "range": "stddev: 0.0007962053039007109",
            "extra": "mean: 32.35288139999852 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.48646857763906,
            "unit": "iter/sec",
            "range": "stddev: 0.020863033238656636",
            "extra": "mean: 95.36098759999732 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.6529247149234,
            "unit": "iter/sec",
            "range": "stddev: 0.009223274362946123",
            "extra": "mean: 10.346298396551244 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3305.634132780921,
            "unit": "iter/sec",
            "range": "stddev: 0.0000074436950505923285",
            "extra": "mean: 302.51381726831727 usec\nrounds: 2189"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.03953742333521,
            "unit": "iter/sec",
            "range": "stddev: 0.00028299862425205686",
            "extra": "mean: 41.598138200001245 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.749027042678414,
            "unit": "iter/sec",
            "range": "stddev: 0.012038228777319972",
            "extra": "mean: 53.33610099999856 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "me@sirmarcel.com",
            "name": "Marcel",
            "username": "sirmarcel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7abb59c29b919f8c5b5999431de9b8e1627db4c6",
          "message": "fix(interpret): shape-aware broadcasting, multi-dim scatter, const propagation (#51)\n\n* fix(interpret): propagate const_vals through convert_element_type\n\nJAX inserts convert_element_type (e.g. int64 → int32) before every\ngather/scatter operation. Without const value propagation through\nthis primitive, the chain from closure-captured index arrays to\ngather/scatter handlers breaks, causing all gathers to fall back\nto conservative (fully dense) index sets.\n\nAlso handles stop_gradient and bitcast_convert_type, which share\nthe same handler but lack the new_dtype parameter.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): handle multi-dimensional scatter patterns\n\nExtend the scatter handler beyond the 1D-only case to support two\nadditional patterns:\n\n1. Batched scatter along dim 0 with trailing window dims preserved.\n   This is the pattern produced by the backward of features[indices]\n   on multi-dimensional arrays (scatter-add with update_window_dims=(1,),\n   inserted_window_dims=(0,)).\n\n2. Full-window scatter along an arbitrary single dimension.\n   This is the pattern from x.at[:, idx, :].set(value) where all update\n   dims are window dims and the scatter targets one position along an\n   arbitrary dimension.\n\nWithout these patterns, any scatter on multi-dimensional arrays fell\nback to conservative (fully dense) index sets.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): use shape-aware broadcasting in binary elementwise ops\n\nThe previous implementation used flat modular indexing (i % len) to\nhandle broadcasting, which only works correctly for scalar (len=1)\nbroadcast. For multi-dimensional broadcasting like (16,16) * (16,1),\nit incorrectly maps output element [p,d] to in2[d] instead of in2[p]\nvia (p*16 + d) % 16 = d.\n\nUse numpy coordinate mapping (np.indices + ravel_multi_index) when\ninput shapes differ, respecting numpy broadcasting rules where size-1\ndimensions always read index 0. The fast path with modular indexing\nis preserved for the common cases of same-shape or scalar inputs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* style: fix lint and formatting\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(benchmarks): add SchNet-style GNN Hessian benchmark\n\nAdds a minimal message-passing GNN as a fourth benchmark group\nexercising gather (neighbor lookup), broadcast (cutoff weighting),\nand scatter-add (gradient of gather) — the patterns pathological\nfor sparsity detection in graph neural networks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): strengthen broadcast and scatter test assertions\n\nAdd test_binary_broadcast_dependent_operands that catches the flat\nmodular indexing bug by using input-dependent operands on both sides.\nUpdate test_scatter_2d to assert the exact conservative pattern with\na TODO documenting the true sparse result.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>\nCo-authored-by: adrhill <adrian.hill@mailbox.org>",
          "timestamp": "2026-02-18T21:29:10+01:00",
          "tree_id": "e18f7f37836df89f9b2a856397ce8ef6a8b1bef2",
          "url": "https://github.com/adrhill/asdex/commit/7abb59c29b919f8c5b5999431de9b8e1627db4c6"
        },
        "date": 1771446593490,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 801.3234340829583,
            "unit": "iter/sec",
            "range": "stddev: 0.004473012612609976",
            "extra": "mean: 1.247935549450652 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3404.027838734071,
            "unit": "iter/sec",
            "range": "stddev: 0.000013354211350070177",
            "extra": "mean: 293.7696303834846 usec\nrounds: 2113"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.331140471314924,
            "unit": "iter/sec",
            "range": "stddev: 0.00026886058424906796",
            "extra": "mean: 17.14350160000322 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 85.81231966963263,
            "unit": "iter/sec",
            "range": "stddev: 0.0003430220766544421",
            "extra": "mean: 11.653338400009261 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.22945221571928,
            "unit": "iter/sec",
            "range": "stddev: 0.01911509782516312",
            "extra": "mean: 47.10437131578709 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 264.3864074775911,
            "unit": "iter/sec",
            "range": "stddev: 0.000034292143476003244",
            "extra": "mean: 3.7823427064220696 msec\nrounds: 218"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.337477928258593,
            "unit": "iter/sec",
            "range": "stddev: 0.0014302339892949402",
            "extra": "mean: 36.57981920001134 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.942281555884037,
            "unit": "iter/sec",
            "range": "stddev: 0.036136918766337296",
            "extra": "mean: 100.58053520000954 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 80.4429006992926,
            "unit": "iter/sec",
            "range": "stddev: 0.017362104570603294",
            "extra": "mean: 12.431177783334135 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3292.1814424296067,
            "unit": "iter/sec",
            "range": "stddev: 0.000014272415786561456",
            "extra": "mean: 303.74996563433854 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.580019393643322,
            "unit": "iter/sec",
            "range": "stddev: 0.00039915045528182756",
            "extra": "mean: 39.09301180000284 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.07715300592426,
            "unit": "iter/sec",
            "range": "stddev: 0.017346199813022725",
            "extra": "mean: 52.418723049999016 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e30a9ad566853c81c1b634b4014e63760c9a4b73",
          "message": "fix(interpret): harden handlers with precise patterns (#52)\n\n* skills: harden `add-handler` skill based on PR #51 post-mortem\n\nAdd requirements to catch the class of bugs found in PR #51:\nconst propagation, broadcasting/non-square shapes, precision\nverification against jax.jacobian, conservative audit, and\ncomposition tests for the const chain.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat(interpret): harden handlers with precise patterns and tests\n\nPost-mortem on PR #51 revealed handlers that silently fell back to\nconservative for untested shapes. This hardens all five at-risk\nhandlers:\n\n- `_gather`: generalize to arbitrary dim + multi-dim collapse\n- `_scatter`: add multi-index pattern (`mat.at[rows, cols].set`)\n- `_conv`: add `feature_group_count` support, explicit batch_group fallback\n- `_concatenate`: propagate `const_vals` for downstream index resolution\n- `_dot_general`, `_pad`: no handler changes, tests only\n\nAdds ~65 new tests covering precision verification against\n`jax.jacobian`, non-square/broadcasting shapes, conservative audits,\nconst chain composition, and edge cases.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* chore: remove redundant TODO.md\n\nTracked in .plan/ files and inline TODO comments instead.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: emphasize documenting conservative patterns with TODO notes\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-18T22:37:16+01:00",
          "tree_id": "099562dcfa6a04d641eba4b5285a640270d2b198",
          "url": "https://github.com/adrhill/asdex/commit/e30a9ad566853c81c1b634b4014e63760c9a4b73"
        },
        "date": 1771450744523,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 872.3550057660034,
            "unit": "iter/sec",
            "range": "stddev: 0.0029335394988741154",
            "extra": "mean: 1.1463223038674641 msec\nrounds: 181"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3304.9614632293815,
            "unit": "iter/sec",
            "range": "stddev: 0.000006140969304036674",
            "extra": "mean: 302.57538888906396 usec\nrounds: 2034"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.01338983405386,
            "unit": "iter/sec",
            "range": "stddev: 0.0004069893390734167",
            "extra": "mean: 17.53973940000151 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78.6683103938378,
            "unit": "iter/sec",
            "range": "stddev: 0.00041383847120407845",
            "extra": "mean: 12.711598799995727 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.993370177403936,
            "unit": "iter/sec",
            "range": "stddev: 0.018540773220447847",
            "extra": "mean: 50.01658005263054 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 246.7515199867785,
            "unit": "iter/sec",
            "range": "stddev: 0.00010669067915464349",
            "extra": "mean: 4.052659939252176 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.641446019788606,
            "unit": "iter/sec",
            "range": "stddev: 0.0006179811480704676",
            "extra": "mean: 36.17755740000348 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.58663334332417,
            "unit": "iter/sec",
            "range": "stddev: 0.03005802880551137",
            "extra": "mean: 104.31190640000523 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 84.239925543096,
            "unit": "iter/sec",
            "range": "stddev: 0.015067889444766652",
            "extra": "mean: 11.870855696428809 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3344.78416465913,
            "unit": "iter/sec",
            "range": "stddev: 0.00000984493894661348",
            "extra": "mean: 298.97295334209133 usec\nrounds: 2229"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.79437688616623,
            "unit": "iter/sec",
            "range": "stddev: 0.00020778867396821323",
            "extra": "mean: 43.87046880000014 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.357844053582088,
            "unit": "iter/sec",
            "range": "stddev: 0.017645249504226454",
            "extra": "mean: 57.61084135294053 msec\nrounds: 17"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b0bfd969dcc656852f3640911d6ecc492ff8189c",
          "message": "test: mark conservative sparsity patterns with fallback TODOs (#53)\n\nAudit all test files for overly conservative sparsity patterns\nand document precise expected results as TODO comments.\n\nFiles edited:\n- test_detection.py: multiply-by-zero (elementwise)\n- test_scalar.py: multiply-by-zero Hessian\n- test_gather.py: dynamic indices fallback\n- test_scatter.py: dynamic indices fallback\n- test_dynamic_slice.py: dynamic start fallback (2 tests)\n- test_dot_general.py: constant matrix zeros ignored\n- test_internals.py: eye @ x gives dense pattern\n- test_platform_index.py: diag via dynamic_update_slice\n- test_scan.py: deps merged across time steps (9 tests)\n- TODO.md: consolidated roadmap of all findings\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-18T23:55:36+01:00",
          "tree_id": "b3eec09525f0fe6c930d09af901e44950e75813e",
          "url": "https://github.com/adrhill/asdex/commit/b0bfd969dcc656852f3640911d6ecc492ff8189c"
        },
        "date": 1771455395524,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 751.0773195559525,
            "unit": "iter/sec",
            "range": "stddev: 0.005351251542570341",
            "extra": "mean: 1.3314208457142789 msec\nrounds: 175"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3281.0376203929595,
            "unit": "iter/sec",
            "range": "stddev: 0.000008899686722998263",
            "extra": "mean: 304.7816318181177 usec\nrounds: 1980"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.307145923735796,
            "unit": "iter/sec",
            "range": "stddev: 0.00028495373566714434",
            "extra": "mean: 17.449830799998267 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 79.06357981577938,
            "unit": "iter/sec",
            "range": "stddev: 0.00021345568981382388",
            "extra": "mean: 12.64804860000055 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.826046774546683,
            "unit": "iter/sec",
            "range": "stddev: 0.01583780482649812",
            "extra": "mean: 48.016794105263735 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.81001918858271,
            "unit": "iter/sec",
            "range": "stddev: 0.00005499882378093921",
            "extra": "mean: 4.00304200467274 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.8888020140331,
            "unit": "iter/sec",
            "range": "stddev: 0.00044473371330233923",
            "extra": "mean: 35.856685399997446 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.031338939619024,
            "unit": "iter/sec",
            "range": "stddev: 0.0007342707566974999",
            "extra": "mean: 90.6508271999968 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 115.06762780771484,
            "unit": "iter/sec",
            "range": "stddev: 0.0003437455676104372",
            "extra": "mean: 8.690541545455877 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3303.1923826390694,
            "unit": "iter/sec",
            "range": "stddev: 0.00001482501621781785",
            "extra": "mean: 302.7374382599705 usec\nrounds: 2138"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.81542322610094,
            "unit": "iter/sec",
            "range": "stddev: 0.0008605542979479484",
            "extra": "mean: 43.829999999999814 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.528880489930103,
            "unit": "iter/sec",
            "range": "stddev: 0.00034191939237370405",
            "extra": "mean: 53.96980138888966 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e8dbf8d1a5c6256c132aba7d643536eca9dc9f38",
          "message": "feat(coloring): warn when coloring yields no speedup over dense AD (#54)\n\nEmit a DenseColoringWarning when color_jacobian_pattern or\ncolor_hessian_pattern produces as many colors as the dense baseline,\nsince sparse differentiation offers no performance benefit in that case.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T00:13:50+01:00",
          "tree_id": "9ef80f0beeb4d6e7a0df654ed472657b97ce2b48",
          "url": "https://github.com/adrhill/asdex/commit/e8dbf8d1a5c6256c132aba7d643536eca9dc9f38"
        },
        "date": 1771456473444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 884.4956460172849,
            "unit": "iter/sec",
            "range": "stddev: 0.0026486268575055547",
            "extra": "mean: 1.130587815217417 msec\nrounds: 184"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3297.9303940879768,
            "unit": "iter/sec",
            "range": "stddev: 0.000008832199711216914",
            "extra": "mean: 303.2204687499307 usec\nrounds: 1952"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 40.77742952312364,
            "unit": "iter/sec",
            "range": "stddev: 0.0170371099112675",
            "extra": "mean: 24.523370200000727 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.34650624865192,
            "unit": "iter/sec",
            "range": "stddev: 0.0004214410016286995",
            "extra": "mean: 11.998103399999138 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.922371688811516,
            "unit": "iter/sec",
            "range": "stddev: 0.012930276512139398",
            "extra": "mean: 45.61550247368392 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.94822186966275,
            "unit": "iter/sec",
            "range": "stddev: 0.00004339211380217952",
            "extra": "mean: 4.01689954838702 msec\nrounds: 217"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.40473321427657,
            "unit": "iter/sec",
            "range": "stddev: 0.0008342170834765767",
            "extra": "mean: 32.88961599999993 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.463897999060581,
            "unit": "iter/sec",
            "range": "stddev: 0.02287484335432458",
            "extra": "mean: 95.56668080000179 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 94.45369400194704,
            "unit": "iter/sec",
            "range": "stddev: 0.010573741783773957",
            "extra": "mean: 10.587198421052609 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3269.446607711375,
            "unit": "iter/sec",
            "range": "stddev: 0.00000878377379941122",
            "extra": "mean: 305.86215955978065 usec\nrounds: 2181"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.587115091848887,
            "unit": "iter/sec",
            "range": "stddev: 0.0003198796690569107",
            "extra": "mean: 40.67170939999869 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.081877646435583,
            "unit": "iter/sec",
            "range": "stddev: 0.012468107953978656",
            "extra": "mean: 52.40574426315934 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "3bb1ecfe1bc47293e69e6696a01ad0475c28adff",
          "message": "docs: update `CLAUDE.md` files\n\nReplace per-file tree listings in `_interpret`, `tests`, and `docs`\n`CLAUDE.md` with naming conventions that won't go stale. Add missing\nmodules (`verify.py`, `_display.py`, `_pad.py`, `_sort.py`) to root\n`CLAUDE.md` and split the architecture diagram into separate Jacobian\nand Hessian sections.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T11:32:13+01:00",
          "tree_id": "613a60c6674af76a4a831cc0f0f3d049ab7b9e4f",
          "url": "https://github.com/adrhill/asdex/commit/3bb1ecfe1bc47293e69e6696a01ad0475c28adff"
        },
        "date": 1771497182206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 798.5158540722431,
            "unit": "iter/sec",
            "range": "stddev: 0.00421201749829527",
            "extra": "mean: 1.2523232881354014 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3351.668446662795,
            "unit": "iter/sec",
            "range": "stddev: 0.00002488966085476916",
            "extra": "mean: 298.3588669087137 usec\nrounds: 1931"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.18611959546998,
            "unit": "iter/sec",
            "range": "stddev: 0.000590905820356176",
            "extra": "mean: 16.895853400001215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.66840927274151,
            "unit": "iter/sec",
            "range": "stddev: 0.0002784418322299438",
            "extra": "mean: 12.096519199985778 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.626501495559452,
            "unit": "iter/sec",
            "range": "stddev: 0.012772713965640347",
            "extra": "mean: 46.23956399999921 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.59552810014853,
            "unit": "iter/sec",
            "range": "stddev: 0.00003513028564607417",
            "extra": "mean: 4.07173537619228 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.052812254406376,
            "unit": "iter/sec",
            "range": "stddev: 0.0008359552020113034",
            "extra": "mean: 34.420075800005634 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.699915837710867,
            "unit": "iter/sec",
            "range": "stddev: 0.03075994940635435",
            "extra": "mean: 103.09367799999336 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 87.54861437761215,
            "unit": "iter/sec",
            "range": "stddev: 0.01369740126866841",
            "extra": "mean: 11.422225321429178 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3285.9717448275483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072548857494248426",
            "extra": "mean: 304.32398013589165 usec\nrounds: 2215"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.49107221312907,
            "unit": "iter/sec",
            "range": "stddev: 0.00041541869706479036",
            "extra": "mean: 40.83120539997935 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.71102102251398,
            "unit": "iter/sec",
            "range": "stddev: 0.013608529684565072",
            "extra": "mean: 53.444437842101344 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ed9d21b04d55f13a05bc74c61e55846e6dce753d",
          "message": "test(interpret): harden non-symmetric shape coverage and inline expected patterns (#55)\n\n* test(interpret): use non-square shapes to catch dimension transposition bugs\n\nReplace square and repeated-size dimensions in handler tests with\nasymmetric shapes so that axis-confusion bugs cannot hide behind\nmatching dimension sizes.\n\nAlso add \"Writing handler tests\" section to tests/CLAUDE.md\ndocumenting the non-square shape guideline and other test conventions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: suppress `DenseColoringWarning` in small Jacobian-value tests\n\nThese tests intentionally use small inputs where coloring cannot\nbeat dense AD. The warning is expected, not a sign of conservatism.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(gather): add fallback tests for `const_vals` through `slice`, `transpose`, `tile`\n\nCover all four shape-transforming ops from TODO #1 that don't propagate\n`const_vals` (`reshape`, `slice`, `transpose`, `tile`). Convert the existing\n`reshape` xfail test to assert the conservative pattern directly, avoiding\nunnecessary Jacobian computation at test time.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): replace `_check_precision` helpers with explicit patterns\n\nInline all `_check_precision` and `_check_sparser_than_conservative`\ncalls in `test_gather.py` and `test_scatter.py` with pre-computed\nexpected sparsity patterns, avoiding runtime Jacobian computation.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T15:30:26+01:00",
          "tree_id": "e3c2d692eea9aae41aa3c1471e132d2af8eee0f6",
          "url": "https://github.com/adrhill/asdex/commit/ed9d21b04d55f13a05bc74c61e55846e6dce753d"
        },
        "date": 1771511469857,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 849.7821333284803,
            "unit": "iter/sec",
            "range": "stddev: 0.0031610397775889924",
            "extra": "mean: 1.176772211111496 msec\nrounds: 180"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3317.5394962346672,
            "unit": "iter/sec",
            "range": "stddev: 0.00001288833276385827",
            "extra": "mean: 301.42821242519574 usec\nrounds: 1996"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.90704617338086,
            "unit": "iter/sec",
            "range": "stddev: 0.0005250805392087143",
            "extra": "mean: 17.269055600002048 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.95340337609102,
            "unit": "iter/sec",
            "range": "stddev: 0.0004045805496032268",
            "extra": "mean: 11.911369400002059 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.692004637998284,
            "unit": "iter/sec",
            "range": "stddev: 0.0007522056222313504",
            "extra": "mean: 42.20833210526035 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.83505731755147,
            "unit": "iter/sec",
            "range": "stddev: 0.00015950712656648698",
            "extra": "mean: 4.034941669768286 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.890910015248863,
            "unit": "iter/sec",
            "range": "stddev: 0.000917704085936548",
            "extra": "mean: 33.45498679999537 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.18143506663311,
            "unit": "iter/sec",
            "range": "stddev: 0.025774078515908308",
            "extra": "mean: 98.21798140001192 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.33815689646414,
            "unit": "iter/sec",
            "range": "stddev: 0.011331668815909409",
            "extra": "mean: 10.71373201754193 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3233.095414329105,
            "unit": "iter/sec",
            "range": "stddev: 0.00004977156727196325",
            "extra": "mean: 309.3011098800215 usec\nrounds: 2166"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.682100293533715,
            "unit": "iter/sec",
            "range": "stddev: 0.0011372360346340676",
            "extra": "mean: 44.08762799999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.29457246209902,
            "unit": "iter/sec",
            "range": "stddev: 0.019756210719187612",
            "extra": "mean: 57.82160861111171 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c76f7134924f89cb8ca5e3228fe62d1340baa076",
          "message": "docs: add explanation pages and improve cross-references (#56)\n\n* docs(explanation): add ASD overview and global sparsity pages\n\nAdd two new explanation pages that provide the conceptual foundation\nfor the existing sparsity detection and graph coloring pages:\n- `asd.md`: AD basics, structural orthogonality, seed matrices,\n  compression/decompression, amortization, and Hessian extension\n- `global-sparsity.md`: local vs. global patterns, conservatism,\n  and the precision-over-speed design decision\n\nMove overlapping content out of `sparsity-detection.md` and\n`coloring.md` into the new pages and update nav + links.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(explanation): expand coloring and sparsity detection pages\n\nColoring: add \"Row and Column Coloring\" and \"From Coloring to\nDecompression\" sections, attribute SparseMatrixColorings.jl in\nreferences, and link to API docs.\n\nSparsity detection: add sections on abstract interpretation motivation,\njaxpr representation, index set propagation with worked example,\nprimitive handler families, and fallback handlers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(explanation): reduce bold formatting to first-use definitions\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(explanation): revise ASD overview and global sparsity pages\n\nUse `c` for number of colors, restructure Hessian section to separate\nthe definition-based argument from the symmetry-based one, and add a\ntip admonition encouraging users to report conservative patterns.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add cross-references and contribution tips across pages\n\nAdd links between explanation, how-to, tutorial, and reference pages\nwhere they help the reader navigate. Add tip admonitions encouraging\nusers to report incorrect or overly dense patterns. Move the \"verify\ncorrectness\" admonitions above the first code block in the how-to\nguides.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T18:33:02+01:00",
          "tree_id": "952f89e921ff1b66bec803f366c1b01ac4037b9c",
          "url": "https://github.com/adrhill/asdex/commit/c76f7134924f89cb8ca5e3228fe62d1340baa076"
        },
        "date": 1771522425826,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 882.1548986779286,
            "unit": "iter/sec",
            "range": "stddev: 0.0028373623623867356",
            "extra": "mean: 1.1335877650270763 msec\nrounds: 183"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3291.1961086860033,
            "unit": "iter/sec",
            "range": "stddev: 0.000017116509600791605",
            "extra": "mean: 303.8409037251949 usec\nrounds: 2067"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.42403656641067,
            "unit": "iter/sec",
            "range": "stddev: 0.00043975286622609743",
            "extra": "mean: 16.82820720000109 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.12014409108957,
            "unit": "iter/sec",
            "range": "stddev: 0.0003672919011332168",
            "extra": "mean: 11.887758999998255 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.396285294739663,
            "unit": "iter/sec",
            "range": "stddev: 0.01403423495561684",
            "extra": "mean: 46.73708478947291 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.1319145331875,
            "unit": "iter/sec",
            "range": "stddev: 0.00004941991824181203",
            "extra": "mean: 4.013937764150997 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.114181154280576,
            "unit": "iter/sec",
            "range": "stddev: 0.0010870628057511403",
            "extra": "mean: 32.13968559999927 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.372656694455081,
            "unit": "iter/sec",
            "range": "stddev: 0.025501523876671754",
            "extra": "mean: 96.40731680000272 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.73807499938631,
            "unit": "iter/sec",
            "range": "stddev: 0.012160783683358165",
            "extra": "mean: 10.668023639343424 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3166.15074011007,
            "unit": "iter/sec",
            "range": "stddev: 0.00006986948186713662",
            "extra": "mean: 315.8409318076989 usec\nrounds: 2185"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.189375107475563,
            "unit": "iter/sec",
            "range": "stddev: 0.001255299686810179",
            "extra": "mean: 39.699277799996935 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.324277820702118,
            "unit": "iter/sec",
            "range": "stddev: 0.011281365271957652",
            "extra": "mean: 49.2022402380964 msec\nrounds: 21"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c0c090b7879293c25257b17b84ca2323651b45fe",
          "message": "feat(pattern): add save/load for `SparsityPattern` and `ColoredPattern` (#57)\n\nSerialize patterns to `.npz` files for reuse across sessions.\nAdd roundtrip tests and document `save`/`load` in the how-to guides.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T18:55:07+01:00",
          "tree_id": "f62143cd5178fc75e8f41d5d63b8c39d9690ddcd",
          "url": "https://github.com/adrhill/asdex/commit/c0c090b7879293c25257b17b84ca2323651b45fe"
        },
        "date": 1771523749365,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 818.0080670852859,
            "unit": "iter/sec",
            "range": "stddev: 0.00394124037437695",
            "extra": "mean: 1.222481831460642 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3058.3166302886725,
            "unit": "iter/sec",
            "range": "stddev: 0.00005998960565478144",
            "extra": "mean: 326.9772626209768 usec\nrounds: 1961"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.152149843107516,
            "unit": "iter/sec",
            "range": "stddev: 0.0005281605586323503",
            "extra": "mean: 17.19626880000078 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.15132365827267,
            "unit": "iter/sec",
            "range": "stddev: 0.00042472096397308556",
            "extra": "mean: 12.172658400000103 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.408668286680765,
            "unit": "iter/sec",
            "range": "stddev: 0.017191695028135303",
            "extra": "mean: 48.99878747368472 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.91939108328776,
            "unit": "iter/sec",
            "range": "stddev: 0.00005880460522324059",
            "extra": "mean: 4.116591909524169 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.460122283545928,
            "unit": "iter/sec",
            "range": "stddev: 0.0010227344767580174",
            "extra": "mean: 33.94419040000116 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.215141215832125,
            "unit": "iter/sec",
            "range": "stddev: 0.0006967969009466868",
            "extra": "mean: 89.16517239999848 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 124.39467160592126,
            "unit": "iter/sec",
            "range": "stddev: 0.00015238817878864767",
            "extra": "mean: 8.03892953846103 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3291.6772976725583,
            "unit": "iter/sec",
            "range": "stddev: 0.000008502756592987899",
            "extra": "mean: 303.7964871912166 usec\nrounds: 2186"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.271863096210154,
            "unit": "iter/sec",
            "range": "stddev: 0.00020535079843341978",
            "extra": "mean: 42.97034559999844 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.80996477284918,
            "unit": "iter/sec",
            "range": "stddev: 0.01666117689111819",
            "extra": "mean: 56.148342388889695 msec\nrounds: 18"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "64b97dc1add960dd9c0d38f7bf66aca326a29c38",
          "message": "feat(decompression): add `hvp_mode` parameter to `hessian()` (#58)\n\nSupport three AD composition strategies for Hessian-vector products:\n`fwd_over_rev` (default), `rev_over_fwd`, and `rev_over_rev`.\nAll produce the same result but differ in memory/performance\ncharacteristics.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T19:28:48+01:00",
          "tree_id": "71bf45f0baa40e41b025b3e5b1d7c78bdedfa7e9",
          "url": "https://github.com/adrhill/asdex/commit/64b97dc1add960dd9c0d38f7bf66aca326a29c38"
        },
        "date": 1771525773744,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 771.410192440663,
            "unit": "iter/sec",
            "range": "stddev: 0.004504040021748754",
            "extra": "mean: 1.2963271807909384 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3352.7207515154037,
            "unit": "iter/sec",
            "range": "stddev: 0.000021807882723399147",
            "extra": "mean: 298.26522222228255 usec\nrounds: 1998"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 55.39888013601172,
            "unit": "iter/sec",
            "range": "stddev: 0.00020892651752188528",
            "extra": "mean: 18.0509064000006 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.99214207756651,
            "unit": "iter/sec",
            "range": "stddev: 0.00026537025967727866",
            "extra": "mean: 12.346876800002349 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.719319932600254,
            "unit": "iter/sec",
            "range": "stddev: 0.0005439059878447428",
            "extra": "mean: 44.015401999999426 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.6778234152053,
            "unit": "iter/sec",
            "range": "stddev: 0.00003814032216731452",
            "extra": "mean: 4.037503181395483 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.887462862495596,
            "unit": "iter/sec",
            "range": "stddev: 0.000555566179971964",
            "extra": "mean: 34.617093399998566 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.716783190751729,
            "unit": "iter/sec",
            "range": "stddev: 0.03109141252410166",
            "extra": "mean: 102.91471780000023 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 82.66905669941337,
            "unit": "iter/sec",
            "range": "stddev: 0.015624223047501802",
            "extra": "mean: 12.096424465517048 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3283.4065366828754,
            "unit": "iter/sec",
            "range": "stddev: 0.000027074195435156823",
            "extra": "mean: 304.5617375819289 usec\nrounds: 2134"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.78740062076028,
            "unit": "iter/sec",
            "range": "stddev: 0.0007957688054684066",
            "extra": "mean: 42.03906159999917 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.230036331349755,
            "unit": "iter/sec",
            "range": "stddev: 0.014206668939813377",
            "extra": "mean: 54.85452589473582 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "18f12d46d7e0b4f42958d5d784c84b2f55d6557c",
          "message": "feat(interpret): improve sparsity via const-value tracking (#59)\n\n* feat(interpret): improve sparsity for const_vals, mul-by-zero, and dot_general\n\nPropagate const_vals through reshape, slice, transpose, and tile so that\nconstant index arrays survive shape-transforming ops before gather/scatter.\n\nAdd prop_mul handler in _mul.py that clears dependencies at positions\nwhere a known-zero constant makes the product zero (d(0*y)/dy = 0).\n\nMake dot_general skip contraction terms where either factor is a known\nconstant zero, producing sparser patterns for matmul with sparse constants.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract `propagate_const_unary` and `propagate_const_binary` into `_commons`\n\nMove const-value propagation utilities to `_commons.py` alongside\n`atom_const_val`. Replace inline const propagation in reshape, slice,\ntranspose, and tile with `propagate_const_unary` calls using `partial`,\n`itemgetter`, or named closures. Remove unnecessary `np.asarray` and\n`.ravel()` round-trips — const values are stored in natural shape.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: remove resolved items from TODO.md\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): make `propagate_const_binary` take a callable transform\n\nSymmetric with `propagate_const_unary`. The ufunc dict lookup\nmoves from `_commons.py` into `propagate_const_elementwise` in\n`_elementwise.py`, keeping the shared utility interface clean.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract `copy_index_sets` utility into `_commons`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): explain downstream impact in `propagate_const_*` docstrings\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): document conservative fallback invariant for const_vals\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-20T15:52:27+01:00",
          "tree_id": "1fb12c5b3c013bef6a21f2d9113078ec71bdc9c8",
          "url": "https://github.com/adrhill/asdex/commit/18f12d46d7e0b4f42958d5d784c84b2f55d6557c"
        },
        "date": 1771599190062,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 792.37178871351,
            "unit": "iter/sec",
            "range": "stddev: 0.0033379364365099056",
            "extra": "mean: 1.2620338258427826 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3306.331088276839,
            "unit": "iter/sec",
            "range": "stddev: 0.000006561970352595943",
            "extra": "mean: 302.45004910296814 usec\nrounds: 2118"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 39.199111967639055,
            "unit": "iter/sec",
            "range": "stddev: 0.019580880003549957",
            "extra": "mean: 25.51078199999921 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.15790203723284,
            "unit": "iter/sec",
            "range": "stddev: 0.00040215842762637903",
            "extra": "mean: 12.02531539999967 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.033269737508014,
            "unit": "iter/sec",
            "range": "stddev: 0.016096975600604108",
            "extra": "mean: 47.54372536842093 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.41569654513728,
            "unit": "iter/sec",
            "range": "stddev: 0.0003605956600528783",
            "extra": "mean: 4.108198502369658 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.34087199022872,
            "unit": "iter/sec",
            "range": "stddev: 0.0008961091991275546",
            "extra": "mean: 32.958841800000016 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.417753650741876,
            "unit": "iter/sec",
            "range": "stddev: 0.022684410072598304",
            "extra": "mean: 95.9899833999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 90.22501554588302,
            "unit": "iter/sec",
            "range": "stddev: 0.012057938986435375",
            "extra": "mean: 11.08340069491548 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3280.8559837031894,
            "unit": "iter/sec",
            "range": "stddev: 0.000032222706948445385",
            "extra": "mean: 304.79850531911296 usec\nrounds: 2256"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.80666146973708,
            "unit": "iter/sec",
            "range": "stddev: 0.001958743692824652",
            "extra": "mean: 40.31175260000026 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.326580007827673,
            "unit": "iter/sec",
            "range": "stddev: 0.013885459815837517",
            "extra": "mean: 51.742212000000976 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "67fbae7d34492dfafdb27279f2c4e94ee3437dfb",
          "message": "refactor(interpret): introduce `IndexSet` abstraction (#60)\n\n* refactor(interpret): introduce `IndexSet` abstraction for future backend swap\n\nAdd `IndexSet` type alias and factory helpers (`empty_index_set`,\n`singleton_index_set`, `empty_index_sets`, `identity_index_sets`) to\ncentralize index set construction in `_commons.py`.\n\nUpdate all handlers, `detection.py`, and tests to use the new helpers\ninstead of bare `set()` / `{i}` / `[set() for ...]` construction.\nThis confines the `set[int]` dependency to the helpers, enabling a\nfuture swap to `pyroaring.BitMap` by changing only the factory functions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): remove `IndexSets` type alias\n\nReplace `IndexSets` with `list[IndexSet]` everywhere to avoid\nthe visually similar `IndexSet` / `IndexSets` pair.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-20T17:59:00+01:00",
          "tree_id": "e8dee067a23c8ac10d12777e6248158111f15482",
          "url": "https://github.com/adrhill/asdex/commit/67fbae7d34492dfafdb27279f2c4e94ee3437dfb"
        },
        "date": 1771606782243,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 816.7046193432479,
            "unit": "iter/sec",
            "range": "stddev: 0.002809837545300645",
            "extra": "mean: 1.2244328932584572 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3301.02072356661,
            "unit": "iter/sec",
            "range": "stddev: 0.00000907544324414739",
            "extra": "mean: 302.93660165803 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.38721045586209,
            "unit": "iter/sec",
            "range": "stddev: 0.0004822521313502439",
            "extra": "mean: 16.838642400003323 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.79039316998548,
            "unit": "iter/sec",
            "range": "stddev: 0.00027866948615172293",
            "extra": "mean: 11.934542400001646 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.229325359203813,
            "unit": "iter/sec",
            "range": "stddev: 0.015430402982792546",
            "extra": "mean: 47.104652789470656 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 251.42000727762192,
            "unit": "iter/sec",
            "range": "stddev: 0.00003213375204018758",
            "extra": "mean: 3.9774082056078544 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.51636894367823,
            "unit": "iter/sec",
            "range": "stddev: 0.0014175528246474227",
            "extra": "mean: 32.76929840000378 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.626124367429021,
            "unit": "iter/sec",
            "range": "stddev: 0.0010791068421252767",
            "extra": "mean: 86.01318619999745 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 120.27343172451452,
            "unit": "iter/sec",
            "range": "stddev: 0.00012254737225083593",
            "extra": "mean: 8.314388187496746 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3326.5003255073534,
            "unit": "iter/sec",
            "range": "stddev: 0.000006836910326655765",
            "extra": "mean: 300.6162339237052 usec\nrounds: 2146"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 19.923974972953214,
            "unit": "iter/sec",
            "range": "stddev: 0.021347409167883247",
            "extra": "mean: 50.190787799999725 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.232329027935187,
            "unit": "iter/sec",
            "range": "stddev: 0.0008047236874152402",
            "extra": "mean: 49.425847050000016 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2032c886c215412736549a4a88861f6400d8b2a9",
          "message": "perf(interpret): remove unnecessary index set copies (#61)\n\n* perf(interpret): remove unnecessary index set copies\n\nIndex sets stored in `deps` are never mutated by subsequent handlers,\nso defensive `.copy()` calls in `permute_indices`, `conservative_indices`,\nbroadcast, pad, sort, top_k, dynamic_update_slice, and scan tiling\nare unnecessary. Removing them avoids millions of small-set copies.\n\nThe only place that mutates stored sets is `fixed_point_loop` (`|=`\non carry), which now copies carry sets on entry to avoid aliasing bugs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf(interpret): use pyroaring for index sets\n\nSwap the `IndexSet` backend from Python's built-in `set` to Roaring\nbitmaps (`pyroaring.BitMap`). Only `_commons.py` changes — all ~28\nhandler modules already use the factory helpers and compatible ops\n(`|=`, `.copy()`, `|`, `len()`, `bool()`, iteration).\n\nFix bare `set()` in `_top_k.py` to use `empty_index_sets()` factory.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Revert \"perf(interpret): use pyroaring for index sets\"\n\nThis reverts commit 9a3f497052b6f760f6dafaf448501c84dc89c8fd.\n\n* docs(interpret): update IndexSet docstring after pyroaring benchmark\n\npyroaring.BitMap and int bitmasks were benchmarked against set[int].\nset[int] wins for the typical workload (small sparse sets, large universe).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): document index set aliasing invariant\n\nHandlers must not mutate sets from `deps` since copies were removed\nfor performance. The only exception is `fixed_point_loop`, which\nexplicitly copies carry sets before in-place mutation.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-20T22:52:24+01:00",
          "tree_id": "553bbfd621d92cca5e919de63d918a5850616fa3",
          "url": "https://github.com/adrhill/asdex/commit/2032c886c215412736549a4a88861f6400d8b2a9"
        },
        "date": 1771624385035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 850.8536821765543,
            "unit": "iter/sec",
            "range": "stddev: 0.0029358324666130566",
            "extra": "mean: 1.1752902067038329 msec\nrounds: 179"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3347.534544134219,
            "unit": "iter/sec",
            "range": "stddev: 0.000007742304312369538",
            "extra": "mean: 298.727313136251 usec\nrounds: 2162"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.7821239481678,
            "unit": "iter/sec",
            "range": "stddev: 0.00047425032540042834",
            "extra": "mean: 16.45220560000098 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.64201423446232,
            "unit": "iter/sec",
            "range": "stddev: 0.00021792935695950805",
            "extra": "mean: 12.248595400012618 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.56787633773195,
            "unit": "iter/sec",
            "range": "stddev: 0.01830257317128752",
            "extra": "mean: 48.619506631586034 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.69138563686113,
            "unit": "iter/sec",
            "range": "stddev: 0.000028911510398288377",
            "extra": "mean: 4.02104800469526 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.244772410640632,
            "unit": "iter/sec",
            "range": "stddev: 0.0004495523208422888",
            "extra": "mean: 34.19414540002208 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.197554282213167,
            "unit": "iter/sec",
            "range": "stddev: 0.025385138541575928",
            "extra": "mean: 98.06272880000506 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.31658397899372,
            "unit": "iter/sec",
            "range": "stddev: 0.010445384294121231",
            "extra": "mean: 10.38242801694562 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3332.4214939902536,
            "unit": "iter/sec",
            "range": "stddev: 0.000013071366817416024",
            "extra": "mean: 300.0820879961965 usec\nrounds: 2216"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.346947753501585,
            "unit": "iter/sec",
            "range": "stddev: 0.001183554105003659",
            "extra": "mean: 39.45248200000151 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.385802751913477,
            "unit": "iter/sec",
            "range": "stddev: 0.013582004569320702",
            "extra": "mean: 51.58414189999405 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "c47a9456cac794f2e64c28c3dd8ddeba3edf0f26",
          "message": "docs: remove complexity claims\n\nThe O(mn) and O(n²) claims for verification assumed T(f) = O(max(m,n)),\nwhich isn't guaranteed. The O(|V| + |E|) claim for greedy coloring\ndidn't account for star coloring's more expensive neighbor-of-neighbor\nchecks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-21T15:33:43+01:00",
          "tree_id": "8c4dffd2cfc0c221af2f8a3760f65183720ae595",
          "url": "https://github.com/adrhill/asdex/commit/c47a9456cac794f2e64c28c3dd8ddeba3edf0f26"
        },
        "date": 1771684487377,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 844.7940319736151,
            "unit": "iter/sec",
            "range": "stddev: 0.0029504359019137945",
            "extra": "mean: 1.1837204835169008 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3110.998753660923,
            "unit": "iter/sec",
            "range": "stddev: 0.00006596238272598864",
            "extra": "mean: 321.4401802068331 usec\nrounds: 2031"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.338533401547195,
            "unit": "iter/sec",
            "range": "stddev: 0.000457191651394767",
            "extra": "mean: 16.30296560000204 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.62813247649675,
            "unit": "iter/sec",
            "range": "stddev: 0.0003415681688527455",
            "extra": "mean: 11.816401599996595 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.030393358213786,
            "unit": "iter/sec",
            "range": "stddev: 0.0005526983680647108",
            "extra": "mean: 41.61396715789473 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.54109458335913,
            "unit": "iter/sec",
            "range": "stddev: 0.00002066550859356318",
            "extra": "mean: 3.991361184331713 msec\nrounds: 217"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.955694346720144,
            "unit": "iter/sec",
            "range": "stddev: 0.000647209152151131",
            "extra": "mean: 32.30423419999795 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.253434016337211,
            "unit": "iter/sec",
            "range": "stddev: 0.0005734415960136834",
            "extra": "mean: 88.86176419999856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.36042346496323,
            "unit": "iter/sec",
            "range": "stddev: 0.00016631014597554748",
            "extra": "mean: 8.172577142856495 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3305.2895672414784,
            "unit": "iter/sec",
            "range": "stddev: 0.000007218132690236099",
            "extra": "mean: 302.5453533363426 usec\nrounds: 2233"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.547270998876815,
            "unit": "iter/sec",
            "range": "stddev: 0.0014519091322791314",
            "extra": "mean: 40.73772600000041 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.63393102124747,
            "unit": "iter/sec",
            "range": "stddev: 0.01535837268428393",
            "extra": "mean: 53.665541578947725 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "047a2b185b2f53e55b204f7a1937b48a228371ec",
          "message": "feat(verify): add randomized matrix-vector product verification (#62)\n\n* feat(verify): add randomized matvec verification and AD mode selection\n\nReplace the O(n²) dense-only verification with a cheap O(k) randomized\nmatvec default path. Both `check_jacobian_correctness` and\n`check_hessian_correctness` gain `method`, `ad_mode`, `num_probes`, and\n`seed` parameters.\n\n- `method=\"matvec\"` (default) checks via randomized matrix-vector\n  products against `jax.jvp` / HVP references\n- `method=\"dense\"` preserves the original full-matrix comparison\n- `ad_mode` controls the reference AD mode for both paths:\n  Jacobian accepts \"forward\" / \"reverse\",\n  Hessian accepts \"fwd_over_rev\" / \"rev_over_fwd\" / \"rev_over_rev\"\n- Tolerances default to 1e-5 (matvec) or 1e-7 (dense)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat(verify): auto-select forward/reverse based on Jacobian shape\n\nWhen `ad_mode` is not specified, pick the mode that maximizes\ndetection power per probe: forward (JVP) when m >= n,\nreverse (VJP) when m < n.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: update verification how-to guides for new API\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-21T17:54:05+01:00",
          "tree_id": "644803c127208018560db850614293401d23b5dc",
          "url": "https://github.com/adrhill/asdex/commit/047a2b185b2f53e55b204f7a1937b48a228371ec"
        },
        "date": 1771692883596,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 858.2120155152836,
            "unit": "iter/sec",
            "range": "stddev: 0.00284591083532933",
            "extra": "mean: 1.1652132362649161 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3342.9018981078243,
            "unit": "iter/sec",
            "range": "stddev: 0.000008635629963343361",
            "extra": "mean: 299.14129414507437 usec\nrounds: 2152"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.330897532256415,
            "unit": "iter/sec",
            "range": "stddev: 0.0004313517798775312",
            "extra": "mean: 17.143573000004153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.46179405274155,
            "unit": "iter/sec",
            "range": "stddev: 0.00020676027131089758",
            "extra": "mean: 12.428258800002823 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.51532895109647,
            "unit": "iter/sec",
            "range": "stddev: 0.0005572506799835429",
            "extra": "mean: 42.52545231579132 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.30826799805723,
            "unit": "iter/sec",
            "range": "stddev: 0.00002790900360007633",
            "extra": "mean: 4.011098420561779 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.098099457922675,
            "unit": "iter/sec",
            "range": "stddev: 0.0005043319718440923",
            "extra": "mean: 33.22468919999437 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.497921265517627,
            "unit": "iter/sec",
            "range": "stddev: 0.0010883339343599088",
            "extra": "mean: 86.97224280001024 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 123.21532828263939,
            "unit": "iter/sec",
            "range": "stddev: 0.0001670011528503709",
            "extra": "mean: 8.115873357137307 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3361.375335158679,
            "unit": "iter/sec",
            "range": "stddev: 0.000012717556775682645",
            "extra": "mean: 297.4972742675859 usec\nrounds: 2184"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.35023350252607,
            "unit": "iter/sec",
            "range": "stddev: 0.0005257201125525064",
            "extra": "mean: 39.447368399993366 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.097500883274197,
            "unit": "iter/sec",
            "range": "stddev: 0.014398258752199822",
            "extra": "mean: 52.36287229999874 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d11498e51cc45cde5c5d4b8998485e824808f573",
          "message": "feat!: unify AD mode API (#63)\n\n* feat!: unify AD mode API\n\nConsistent `mode` parameter everywhere, replacing the mix of\n`partition`, `hvp_mode`, and `ad_mode`. Rename `HvpMode` to\n`HessianMode` and introduce `JacobianMode` with `\"fwd\"`/`\"rev\"`\nvalues (matching JAX's `jacfwd`/`jacrev` naming). Add validation\nfor unknown mode values in `color_jacobian_pattern`.\n\nBREAKING CHANGE: `partition=\"row\"/\"column\"` is now `mode=\"rev\"/\"fwd\"`,\n`hvp_mode` is now `mode`, `ad_mode` is now `mode`,\n`HvpMode` is renamed to `HessianMode`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat!: separate coloring mode from AD mode\n\nRename `ColoringMode` to domain-appropriate values\n(`\"row\"`, `\"column\"`, `\"symmetric\"`) and add `\"auto\"` to all mode types.\n\nReplace the single `mode` kwarg on `jacobian()` and `hessian()` with\nseparate `coloring_mode` and `ad_mode` kwargs, decoupling the coloring\nstrategy from the AD primitive selection.\n\nKey changes:\n- `modes.py`: add `resolve_ad_mode()` and `resolve_hessian_mode()`\n- `coloring.py`: `color_jacobian_pattern` accepts `\"symmetric\"`\n- `decompression.py`: `hessian()` accepts `coloring_mode` for\n  row/column coloring with HVPs\n- `verify.py`: rename `mode` → `ad_mode`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(coloring): centralize mode validation and resolution\n\nAdd `assert_coloring_mode`, `assert_jacobian_mode`, `assert_hessian_mode`\nvalidators and `resolve_coloring_mode` to `modes.py` as single source of truth.\n\n- Derive valid mode sets from `Literal` types via `get_args`\n- Push coloring-from-AD resolution into `resolve_coloring_mode`\n- Push coloring dispatch into `color_hessian_pattern` (accepts `coloring_mode`)\n- Extract `_empty_colored_pattern` helper for zero-nnz patterns\n- Rename bare `mode` parameters to `coloring_mode` everywhere\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: exhaustive match/case with `assert_never`\n\nReplace `if`/`elif` chains on mode arguments with exhaustive\n`match`/`case` using `assert_never` for unreachable branches.\nValidate `coloring_mode` and `ad_mode` eagerly in `jacobian()`\nand `hessian()` so typos raise at construction time.\n\nSuppresses `ty` false positives where match/case narrowing on\nattribute access is not yet supported (`@Todo`).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: update for new `coloring_mode`/`ad_mode` API\n\nFix parameter names in how-to guides (`mode` → `coloring_mode`/`ad_mode`)\nand update reference pages to reflect the new public API:\nremove `color_rows`, `color_cols`, `color_symmetric` and add\n`ColoringMode`, `JacobianMode`, `HessianMode`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(modes): centralize input validation and mark helpers private\n\nAdd `_assert_jacobian_args` and `_assert_hessian_args` to centralize\nthe shared validation (mode assertions + coloring_mode-ignored warning)\nused by `jacobian`, `hessian`, `check_jacobian_correctness`, and\n`check_hessian_correctness`.\n\nPrefix all helper functions in `modes.py` with underscore since only\nthe type aliases are part of the public API.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: fix stale mode terminology and add missing mode tests\n\nRename test names and docstrings that still referenced `hvp_mode` or\nbare `mode` to use the new `ad_mode`/`coloring_mode` API.  Add tests\nfor column+rev incompatibility, invalid mode validation, and the\ncoloring_mode-ignored warning.  Also include the minor modes.py\ndocstring fix.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: cover remaining reachable mode validation paths\n\nAdd tests for:\n- Hessian coloring_mode-ignored warning (_assert_hessian_args)\n- _resolve_ad_mode raising on unresolved coloring_mode=\"auto\"\n- Empty non-square pattern with symmetric coloring (_empty_colored_pattern)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: exclude unreachable branches from coverage\n\nExclude `assert_never()`, `case _ as unreachable:`, and\n`if TYPE_CHECKING:` lines from coverage reports via\n`[tool.coverage.report]` in pyproject.toml.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-21T21:48:17+01:00",
          "tree_id": "5e28607067564842f4e4033909344a1dbf9acb23",
          "url": "https://github.com/adrhill/asdex/commit/d11498e51cc45cde5c5d4b8998485e824808f573"
        },
        "date": 1771706937208,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 840.3929546409895,
            "unit": "iter/sec",
            "range": "stddev: 0.0030180598920961886",
            "extra": "mean: 1.1899195423731195 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3253.506438144869,
            "unit": "iter/sec",
            "range": "stddev: 0.000039471492691002934",
            "extra": "mean: 307.3606949953338 usec\nrounds: 2118"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.734561590613104,
            "unit": "iter/sec",
            "range": "stddev: 0.0005078685265224629",
            "extra": "mean: 16.740727199999128 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 77.11610446455897,
            "unit": "iter/sec",
            "range": "stddev: 0.0014559462925232151",
            "extra": "mean: 12.96746000000013 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.51922044937148,
            "unit": "iter/sec",
            "range": "stddev: 0.015784789925303182",
            "extra": "mean: 48.7347948947364 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.74193693615496,
            "unit": "iter/sec",
            "range": "stddev: 0.00004433561243032252",
            "extra": "mean: 3.988164134883526 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.520557344200146,
            "unit": "iter/sec",
            "range": "stddev: 0.00034150235812334856",
            "extra": "mean: 32.764801399999044 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.859623667195766,
            "unit": "iter/sec",
            "range": "stddev: 0.031092635241702502",
            "extra": "mean: 101.4237494000028 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.25429082543451,
            "unit": "iter/sec",
            "range": "stddev: 0.011225842022172068",
            "extra": "mean: 10.38914724137947 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3280.0620544069498,
            "unit": "iter/sec",
            "range": "stddev: 0.000009027701532782302",
            "extra": "mean: 304.87228089372366 usec\nrounds: 2193"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.13158937896158,
            "unit": "iter/sec",
            "range": "stddev: 0.0007454069748235143",
            "extra": "mean: 39.7905594000008 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.641544443978404,
            "unit": "iter/sec",
            "range": "stddev: 0.01646206413038489",
            "extra": "mean: 53.643623950000574 msec\nrounds: 20"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "89a23cb80924fe0f850ffd27f93f7a6e49fc73e3",
          "message": "refactor!(API): require pre-computed coloring (#64)\n\n* feat!: require `colored_pattern` or `input_shape` in `jacobian()`/`hessian()`\n\n`jacobian(f)(x)` and `hessian(f)(x)` previously re-detected sparsity\nand re-colored on every call. Now either a pre-computed `colored_pattern`\nor an `input_shape` must be provided, and detection + coloring happen\nonce at definition time.\n\n- Add `input_shape` kwarg to `jacobian()` and `hessian()`\n- Add mutual exclusivity check in `_assert_jacobian_args`/`_assert_hessian_args`\n- Remove `colored_pattern is None` branch from `_eval_jacobian`/`_eval_hessian`\n- Update all tests and docs\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat!: freeze AD mode in `ColoredPattern`, remove `ColoringMode`\n\nReplace `mode: ColoringMode` on `ColoredPattern` with `symmetric: bool`\n+ `mode: str` (the resolved AD mode, never \"auto\").\nThis eliminates `ColoringMode` from the public API entirely.\n\n`jacobian()` and `hessian()` now require a pre-computed `ColoredPattern`;\nthe `input_shape` convenience path is removed.\n\nKey changes:\n- `ColoredPattern` stores `symmetric` + `mode` (\"fwd\"/\"rev\"/HVP modes)\n- `ColoringMode` type alias removed from `modes.py` and `__init__.py`\n- `jacobian(f, colored_pattern)` / `hessian(f, colored_pattern)` simplified\n- `from_coordinates` renamed to `from_coo` (scipy convention)\n- All coloring functions use `symmetric=` + `mode=` keyword args\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: rename `colored_pattern` to `coloring`, `sparsity_pattern` to `sparsity`\n\nShorter, more consistent variable names throughout\nthe codebase, tests, and documentation.\n\n* refactor(coloring): put `mode` before `symmetric` in keyword args\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update docs\n\n* feat!: add one-call `jacobian`/`hessian` API, rename old to `_from_coloring`\n\n`jacobian(f, input_shape)` and `hessian(f, input_shape)` now handle\ndetection, coloring, and decompression in one call.\nThe old `jacobian(f, coloring)` and `hessian(f, coloring)` are renamed\nto `jacobian_from_coloring` and `hessian_from_coloring`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: use one-call `jacobian`/`hessian` API in tests\n\nSwitch ~35 end-to-end tests from verbose\n`jacobian_from_coloring(f, jacobian_coloring(f, shape))` to the new\n`jacobian(f, input_shape=shape)` (and likewise for `hessian`).\nTests that exercise explicit coloring paths stay with `from_coloring`.\n\nAlso parametrize `test_hessian_non_symmetric_coloring` over all three\nHessian AD modes.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: minor changes to \"Getting started\" guide\n\n* docs: improve how-to guides for jacobians and hessians\n\n- Rewrite Basic Usage sections with cross-references to API docs\n- Add executable code blocks to Precomputing sections showing coloring output\n- Add imports to Saving/Loading code blocks for self-contained examples\n- Inline multi-dimensional input notes, remove standalone sections\n- Replace \"colored pattern\" with \"coloring\" throughout\n- Add one-call API `mode` parameter mention to Choosing Row vs Column Coloring\n- Fix `jnp` import order in Getting Started tutorial\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor!: rename `color_*_pattern` to `*_coloring_from_sparsity`\n\nRename `color_jacobian_pattern` to `jacobian_coloring_from_sparsity`\nand `color_hessian_pattern` to `hessian_coloring_from_sparsity`\nfor consistency with the `*_coloring` naming convention.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor!: make `coloring` mandatory in `check_*_correctness`\n\n`check_jacobian_correctness` and `check_hessian_correctness` now require\na pre-computed `coloring` as a positional argument instead of accepting\nan optional keyword. Remove auto-detection branches, unused imports, and\nthe non-existent `ad_mode` parameter from docs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Minor tweaks to docstrings\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-22T01:21:19+01:00",
          "tree_id": "04c653e55f9fa29ab7878972af995167e180f88e",
          "url": "https://github.com/adrhill/asdex/commit/89a23cb80924fe0f850ffd27f93f7a6e49fc73e3"
        },
        "date": 1771719721006,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 794.767798100254,
            "unit": "iter/sec",
            "range": "stddev: 0.0037189376688620246",
            "extra": "mean: 1.2582291360952413 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3337.9463102185696,
            "unit": "iter/sec",
            "range": "stddev: 0.000010757101011014845",
            "extra": "mean: 299.58540583431966 usec\nrounds: 1954"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.713199889458544,
            "unit": "iter/sec",
            "range": "stddev: 0.0005753496294292766",
            "extra": "mean: 16.746716000000106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 103.69472389428726,
            "unit": "iter/sec",
            "range": "stddev: 0.0002659972911703389",
            "extra": "mean: 9.643692199995257 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.46058089440822,
            "unit": "iter/sec",
            "range": "stddev: 0.012884775017442611",
            "extra": "mean: 44.52244600000349 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.1920289813675,
            "unit": "iter/sec",
            "range": "stddev: 0.000044750004852854335",
            "extra": "mean: 3.9969298944950515 msec\nrounds: 218"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.359460628471936,
            "unit": "iter/sec",
            "range": "stddev: 0.0005372516271521271",
            "extra": "mean: 32.93866160000789 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.22939960453755,
            "unit": "iter/sec",
            "range": "stddev: 0.0007653826011410219",
            "extra": "mean: 36.72501099999863 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 108.72873537756934,
            "unit": "iter/sec",
            "range": "stddev: 0.007540434363582972",
            "extra": "mean: 9.197200689655949 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3363.0657876179353,
            "unit": "iter/sec",
            "range": "stddev: 0.000008840694381832143",
            "extra": "mean: 297.34773660443363 usec\nrounds: 2221"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.377193277854804,
            "unit": "iter/sec",
            "range": "stddev: 0.0005925186228893789",
            "extra": "mean: 39.405461000001196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.5704144985585,
            "unit": "iter/sec",
            "range": "stddev: 0.0010615404415476647",
            "extra": "mean: 39.10769612500312 msec\nrounds: 24"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "87022d28acde3b083ed3501358df2dad61e24681",
          "message": "docs: update README",
          "timestamp": "2026-02-22T01:28:05+01:00",
          "tree_id": "1881e0201626189bc7c5d2a3025994364e607048",
          "url": "https://github.com/adrhill/asdex/commit/87022d28acde3b083ed3501358df2dad61e24681"
        },
        "date": 1771720130303,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 662.4741502201823,
            "unit": "iter/sec",
            "range": "stddev: 0.005329607505662316",
            "extra": "mean: 1.5094928604650255 msec\nrounds: 129"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3283.5649657028052,
            "unit": "iter/sec",
            "range": "stddev: 0.000007633010419312407",
            "extra": "mean: 304.5470427553921 usec\nrounds: 2105"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.03020664917334,
            "unit": "iter/sec",
            "range": "stddev: 0.0004595149700283407",
            "extra": "mean: 17.232404599997153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 100.27799667907357,
            "unit": "iter/sec",
            "range": "stddev: 0.00046881559437838926",
            "extra": "mean: 9.972277399999996 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.35474301816279,
            "unit": "iter/sec",
            "range": "stddev: 0.023258211307440387",
            "extra": "mean: 51.66692211111171 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 246.6799297119202,
            "unit": "iter/sec",
            "range": "stddev: 0.000060774692158124454",
            "extra": "mean: 4.053836082926683 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.02217485870521,
            "unit": "iter/sec",
            "range": "stddev: 0.0008709241216604976",
            "extra": "mean: 33.30871280000025 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.316013767774592,
            "unit": "iter/sec",
            "range": "stddev: 0.0005344329204518354",
            "extra": "mean: 36.608562599998606 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.63387585238205,
            "unit": "iter/sec",
            "range": "stddev: 0.012930935828396764",
            "extra": "mean: 10.67989539999972 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3274.8477902757645,
            "unit": "iter/sec",
            "range": "stddev: 0.000009415037798966306",
            "extra": "mean: 305.35770333185263 usec\nrounds: 2191"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.87003664948106,
            "unit": "iter/sec",
            "range": "stddev: 0.0005747262250399623",
            "extra": "mean: 40.20902799999959 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.62462762193664,
            "unit": "iter/sec",
            "range": "stddev: 0.0007321251787339029",
            "extra": "mean: 39.024957347825946 msec\nrounds: 23"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "00291be56f8baf65bf996047301c9bd73d571b72",
          "message": "test(bench): remove GNN benchmarks (#65)",
          "timestamp": "2026-02-22T01:37:00+01:00",
          "tree_id": "efd4bb01f2c2e9bb87003ee23023a3b0eb7c9f23",
          "url": "https://github.com/adrhill/asdex/commit/00291be56f8baf65bf996047301c9bd73d571b72"
        },
        "date": 1771720651275,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 806.4634435112215,
            "unit": "iter/sec",
            "range": "stddev: 0.00370230500134805",
            "extra": "mean: 1.2399818095239994 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3233.3872675611765,
            "unit": "iter/sec",
            "range": "stddev: 0.000023506775573701505",
            "extra": "mean: 309.273191625531 usec\nrounds: 2030"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.49765138363673,
            "unit": "iter/sec",
            "range": "stddev: 0.0004260410141572529",
            "extra": "mean: 17.09470340000223 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 91.53455987081635,
            "unit": "iter/sec",
            "range": "stddev: 0.0015885966847490622",
            "extra": "mean: 10.924835399998756 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.798257634979436,
            "unit": "iter/sec",
            "range": "stddev: 0.011485861941200022",
            "extra": "mean: 45.875226210525675 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.50795378739969,
            "unit": "iter/sec",
            "range": "stddev: 0.0005560989788646821",
            "extra": "mean: 4.15786664953278 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.811518403958246,
            "unit": "iter/sec",
            "range": "stddev: 0.0008789278677389984",
            "extra": "mean: 33.54408140000089 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.48138130919971,
            "unit": "iter/sec",
            "range": "stddev: 0.0007491633559385123",
            "extra": "mean: 36.38827280000072 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.89074818495584,
            "unit": "iter/sec",
            "range": "stddev: 0.00018164148721632384",
            "extra": "mean: 8.137309071427877 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3308.4727465058995,
            "unit": "iter/sec",
            "range": "stddev: 0.000007888791590333751",
            "extra": "mean: 302.2542655236035 usec\nrounds: 2158"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.79140941964142,
            "unit": "iter/sec",
            "range": "stddev: 0.0012505421160781093",
            "extra": "mean: 40.336552999997366 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.195832138201844,
            "unit": "iter/sec",
            "range": "stddev: 0.0007676094668254782",
            "extra": "mean: 39.68910391666736 msec\nrounds: 24"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "68e9ca88b8f80e35bda3c8c4004a29ecedcddf11",
          "message": "fix!: validate and simplify mode handling (#66)\n\n* refactor!(API): replace `\"auto\"` mode with `None`\n\nRemove `\"auto\"` from `JacobianMode` and `HessianMode` Literal types.\nFunctions now use `mode: XMode | None = None` where `None` means\nauto-select.\n\n- Add `ColoringMode` type for the resolved mode in `ColoredPattern.mode`\n- Replace `_resolve_*` functions with `_assert_*` validation;\n  `None` handling lives at the call sites\n- Replace `case _:` fallbacks with `assert_never`\n- Tighten `str` annotations to `Literal` types\n- Add tests for invalid mode and method strings\n- Document code style conventions in `CLAUDE.md`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix: add `assert_never` to all exhaustive match statements\n\nEnsures exhaustiveness is enforced at both type-check time and\nruntime for match statements in `coloring.py` and `verify.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: cover remaining branches in verify, coloring, and pattern modules\n\nAdd tests for uncovered edge cases:\n- `check_jacobian_correctness` with a Hessian-mode coloring (shape-based fallback)\n- `check_hessian_correctness` with a Jacobian-mode coloring (ValueError)\n- `_check_allclose` shape-mismatch branch (VerificationError)\n- `_color_jacobian_symmetric` with empty square pattern\n- `_empty_hessian_pattern` with non-square symmetric (ValueError)\n- `_greedy_color` with 0 vertices\n- `_parse_coloring_mode` with invalid mode string via corrupted save file\n\nBrings `verify.py`, `coloring.py`, and `pattern.py` to 100% coverage.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix: validate coloring mode before decompression and reject mismatches\n\nAdd runtime guards (`_assert_jacobian_mode`, `_assert_hessian_mode`)\nbefore match statements in `_eval_jacobian`, `_compute_hvps`, and\n`check_jacobian_correctness`.\nWiden assert signatures to `str` so they work on untyped inputs.\nRemove the silent hessian-mode fallback in `check_jacobian_correctness`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: move mode validation from `pattern` to `modes` module\n\nRename `_parse_coloring_mode` to `_assert_coloring_mode` to match\nthe `_assert_jacobian_mode`/`_assert_hessian_mode` helpers.\nDefine `ColoringMode` as `JacobianMode | HessianMode`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix: flatten `get_args` for union-based `ColoringMode`\n\n`get_args` on `JacobianMode | HessianMode` returns the two `Literal`\ntypes, not the string values.\nExpand both in `_assert_coloring_mode`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-22T03:15:29+01:00",
          "tree_id": "61a66d056e2cf15265ef0dcc58a30817dc5d9db7",
          "url": "https://github.com/adrhill/asdex/commit/68e9ca88b8f80e35bda3c8c4004a29ecedcddf11"
        },
        "date": 1771726561924,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 788.3529228832231,
            "unit": "iter/sec",
            "range": "stddev: 0.004015844633508554",
            "extra": "mean: 1.268467422360439 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3315.1734074318,
            "unit": "iter/sec",
            "range": "stddev: 0.000011317824942420281",
            "extra": "mean: 301.6433462449496 usec\nrounds: 2117"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.064876139369005,
            "unit": "iter/sec",
            "range": "stddev: 0.000612504458209593",
            "extra": "mean: 16.930535799997415 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 101.05386447865553,
            "unit": "iter/sec",
            "range": "stddev: 0.00041680588085491106",
            "extra": "mean: 9.895712599998774 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.43030130719188,
            "unit": "iter/sec",
            "range": "stddev: 0.01888874323300431",
            "extra": "mean: 48.946904157893144 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.06436255484095,
            "unit": "iter/sec",
            "range": "stddev: 0.000034063570037424784",
            "extra": "mean: 4.031211858490655 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.44750878480636,
            "unit": "iter/sec",
            "range": "stddev: 0.0009704826261946985",
            "extra": "mean: 35.15246299999717 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 26.805358592757834,
            "unit": "iter/sec",
            "range": "stddev: 0.0005590263457850977",
            "extra": "mean: 37.30597360000161 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 91.32074657048187,
            "unit": "iter/sec",
            "range": "stddev: 0.013567500916693176",
            "extra": "mean: 10.950414199999933 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3267.2706083934104,
            "unit": "iter/sec",
            "range": "stddev: 0.000009285865134925943",
            "extra": "mean: 306.06586348589053 usec\nrounds: 2117"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.537789976307188,
            "unit": "iter/sec",
            "range": "stddev: 0.0015244892745488018",
            "extra": "mean: 39.157656199998314 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.60218623171767,
            "unit": "iter/sec",
            "range": "stddev: 0.002322823161517912",
            "extra": "mean: 39.05916436000041 msec\nrounds: 25"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b973e62bb93c706055194fa8177e37b5e52d3050",
          "message": "test: increase code coverage to 100% (#67)\n\n* test: push code coverage to 100%\n\nReplace unreachable defensive branches with assertions in\n`_gather`, `_scan`, `_scatter`, and `_commons`.\n\nAdd tests covering:\n- `check_no_index_sets` error via input-dependent conv kernel\n- `batch_group_count > 1` conservative fallback in conv\n- scalar dot product zero-skipping in `dot_general`\n- multi-dim gather with kept dims, 1D coordinates, batching dims,\n  and pattern-mismatch fallbacks\n- const propagation through `reshape` with `dimensions`\n- all-const `select_if_vmap` propagation (Equinox)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add fallback markers and TODO comments to conservative tests\n\nMark six tests with `@pytest.mark.fallback` and document the true\nsparsity patterns from `jax.jacobian`:\n- `conv_general_dilated` with `batch_group_count > 1` (64/576 nnz)\n- Five `gather` patterns with unrecognized dimension_numbers\n\nAdd corresponding entries to TODO.md (sections 5–6).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add JAX source permalinks to assertion comments\n\nReplace bare asserts with explicit AssertionError raises and add\npermalink comments to the JAX source where these invariants are\nenforced at trace time.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: fix assertion comments to match actual JAX validation mechanisms\n\nScan: clarify that JAX rejects 0-d arrays via IndexError on shape[0],\nnot an explicit check. Update permalink from L336 to L334.\nScatter: clarify the indirect relationship between JAX's shape validation\nand asdex's flat-index bounds check.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-24T17:37:19+01:00",
          "tree_id": "611299995948666b40b9a5b701304c8a571c888f",
          "url": "https://github.com/adrhill/asdex/commit/b973e62bb93c706055194fa8177e37b5e52d3050"
        },
        "date": 1771951078626,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 880.9980606850563,
            "unit": "iter/sec",
            "range": "stddev: 0.002707396870998843",
            "extra": "mean: 1.1350762783999873 msec\nrounds: 176"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3347.9863383023753,
            "unit": "iter/sec",
            "range": "stddev: 0.000007072497426665018",
            "extra": "mean: 298.6870013654412 usec\nrounds: 2199"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.244768778626465,
            "unit": "iter/sec",
            "range": "stddev: 0.00037101548064503705",
            "extra": "mean: 16.879127400034122 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 104.57113693113632,
            "unit": "iter/sec",
            "range": "stddev: 0.0003011483525232324",
            "extra": "mean: 9.562868200032426 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.117895258419367,
            "unit": "iter/sec",
            "range": "stddev: 0.013743550095311334",
            "extra": "mean: 47.35320389475443 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.22417689215231,
            "unit": "iter/sec",
            "range": "stddev: 0.00003395728920455152",
            "extra": "mean: 3.9964163831818866 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.23112761516873,
            "unit": "iter/sec",
            "range": "stddev: 0.0009150282281053453",
            "extra": "mean: 32.019336999996995 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.87412639909137,
            "unit": "iter/sec",
            "range": "stddev: 0.0007756923707773017",
            "extra": "mean: 35.875563800004784 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 124.06967799446019,
            "unit": "iter/sec",
            "range": "stddev: 0.00016842842513783122",
            "extra": "mean: 8.059987066659838 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3330.2806898214485,
            "unit": "iter/sec",
            "range": "stddev: 0.000012533514670249688",
            "extra": "mean: 300.27498974977226 usec\nrounds: 2244"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.20928106171649,
            "unit": "iter/sec",
            "range": "stddev: 0.001223986733515063",
            "extra": "mean: 41.30647240001508 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.978517783459854,
            "unit": "iter/sec",
            "range": "stddev: 0.001002026558603782",
            "extra": "mean: 38.49334316666386 msec\nrounds: 24"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "15915182a54c990fa860e64b7d4adb8f93aa0773",
          "message": "test: mark slow tests and skip them by default (#68)\n\nAdd `@pytest.mark.slow` to 9 tests that take >1s.\nDefault pytest config now skips `slow` and `benchmark` markers.\nCI overrides with `-m \"\"` to run all tests for full coverage.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-26T13:04:59+01:00",
          "tree_id": "c8be5e0b8f6dac9879da53d54149f69927f98602",
          "url": "https://github.com/adrhill/asdex/commit/15915182a54c990fa860e64b7d4adb8f93aa0773"
        },
        "date": 1772107535023,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 756.6166827694085,
            "unit": "iter/sec",
            "range": "stddev: 0.004627585846744149",
            "extra": "mean: 1.3216732101911195 msec\nrounds: 157"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3335.012357961804,
            "unit": "iter/sec",
            "range": "stddev: 0.000008693895395523788",
            "extra": "mean: 299.8489638614566 usec\nrounds: 2020"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.31823841596443,
            "unit": "iter/sec",
            "range": "stddev: 0.0002690788859913953",
            "extra": "mean: 17.756237199999703 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 97.39172873864193,
            "unit": "iter/sec",
            "range": "stddev: 0.0004203442718397926",
            "extra": "mean: 10.267812399999343 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.15699790726377,
            "unit": "iter/sec",
            "range": "stddev: 0.021875045935741945",
            "extra": "mean: 52.20024582352904 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.6045091137162,
            "unit": "iter/sec",
            "range": "stddev: 0.00008890544190984269",
            "extra": "mean: 3.990351185366072 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.69182026904335,
            "unit": "iter/sec",
            "range": "stddev: 0.00048327270332239905",
            "extra": "mean: 34.85313900000051 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 22.804411343710985,
            "unit": "iter/sec",
            "range": "stddev: 0.006045566423375936",
            "extra": "mean: 43.85116479999738 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 87.04200009982264,
            "unit": "iter/sec",
            "range": "stddev: 0.015614946030558331",
            "extra": "mean: 11.488706588235184 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3280.3622452871637,
            "unit": "iter/sec",
            "range": "stddev: 0.00000817574784615013",
            "extra": "mean: 304.84438157300514 usec\nrounds: 2225"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.261409891360625,
            "unit": "iter/sec",
            "range": "stddev: 0.0007816494760756332",
            "extra": "mean: 42.989655599998855 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 23.190818839994538,
            "unit": "iter/sec",
            "range": "stddev: 0.0017733301361524815",
            "extra": "mean: 43.120512772727764 msec\nrounds: 22"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59bf84fe5bccb6d09ac75fabff36fcc69877b257",
          "message": "refactor(interpret): add `transform_indices` (#69)\n\n* refactor: replace `permute_indices` with `transform_indices`\n\nThe `position_map → transform → .ravel() → permute_indices()` pipeline\nwas repeated across handler files. `transform_indices` absorbs this into\none call, eliminating the `.ravel()` footgun and reducing boilerplate.\n\nCall sites using `np.ravel_multi_index` (tile, broadcast, gather,\ndynamic_update_slice) inline the one-liner instead.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: restore `permute_indices` alongside `transform_indices`\n\nKeep both utilities for their natural use cases:\n- `permute_indices` for handlers with a precomputed flat map\n  (broadcast, tile, gather)\n- `transform_indices` for handlers that transform an ndarray\n  (transpose, rev, slice, reshape, split, dynamic_slice)\n\n`transform_indices` now delegates to `permute_indices` internally,\neliminating raw `[in_indices[j] for j in flat_map]` list comprehensions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-26T15:02:23+01:00",
          "tree_id": "2c23e1d05d4effdb5fb4e249f4005ced52c07516",
          "url": "https://github.com/adrhill/asdex/commit/59bf84fe5bccb6d09ac75fabff36fcc69877b257"
        },
        "date": 1772114581224,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 778.0144825479838,
            "unit": "iter/sec",
            "range": "stddev: 0.004199208434481189",
            "extra": "mean: 1.2853231172831354 msec\nrounds: 162"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3259.2969696735904,
            "unit": "iter/sec",
            "range": "stddev: 0.0000370590470623779",
            "extra": "mean: 306.81463189902183 usec\nrounds: 2138"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.26647943733294,
            "unit": "iter/sec",
            "range": "stddev: 0.0005869992517597614",
            "extra": "mean: 17.462222400004634 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 103.72390389559858,
            "unit": "iter/sec",
            "range": "stddev: 0.00019792618144412572",
            "extra": "mean: 9.640979199997446 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.324084296831316,
            "unit": "iter/sec",
            "range": "stddev: 0.018308252891349738",
            "extra": "mean: 49.20270873684124 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.42861289509926,
            "unit": "iter/sec",
            "range": "stddev: 0.0001378404306884008",
            "extra": "mean: 4.074504550239293 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.411578720898948,
            "unit": "iter/sec",
            "range": "stddev: 0.0011102177545178544",
            "extra": "mean: 34.00021500000037 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 26.191269287054826,
            "unit": "iter/sec",
            "range": "stddev: 0.0009344920682729239",
            "extra": "mean: 38.18066199999919 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.17280563044542,
            "unit": "iter/sec",
            "range": "stddev: 0.012497141794383942",
            "extra": "mean: 10.732745388888846 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3267.477486401105,
            "unit": "iter/sec",
            "range": "stddev: 0.00001510803659360926",
            "extra": "mean: 306.0464851439356 usec\nrounds: 2154"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.101643790533476,
            "unit": "iter/sec",
            "range": "stddev: 0.0008230525480828906",
            "extra": "mean: 43.2869629999999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 22.825949992927328,
            "unit": "iter/sec",
            "range": "stddev: 0.003140215051811267",
            "extra": "mean: 43.80978668181838 msec\nrounds: 22"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59b205875971d2d8bf6eb9c8305056f5a2c9cbc9",
          "message": "ci(release): add PyPI publish workflow (#70)\n\n* ci: add PyPI publish workflow\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* ci(release): use `uv publish` with trusted publishing\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-26T15:02:54+01:00",
          "tree_id": "6532214c3bd68d51fa898ccae95246916b6ae301",
          "url": "https://github.com/adrhill/asdex/commit/59b205875971d2d8bf6eb9c8305056f5a2c9cbc9"
        },
        "date": 1772114728749,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 830.9284097151459,
            "unit": "iter/sec",
            "range": "stddev: 0.0034670622183574757",
            "extra": "mean: 1.2034731130962466 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3299.9375076458264,
            "unit": "iter/sec",
            "range": "stddev: 0.000008402254665318898",
            "extra": "mean: 303.03604164716427 usec\nrounds: 2137"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.86396964864214,
            "unit": "iter/sec",
            "range": "stddev: 0.0005361831943697026",
            "extra": "mean: 16.430081800001517 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 104.48882749026433,
            "unit": "iter/sec",
            "range": "stddev: 0.00029457093183764277",
            "extra": "mean: 9.570401200005563 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.440166234625888,
            "unit": "iter/sec",
            "range": "stddev: 0.011017483621374531",
            "extra": "mean: 44.56294973684144 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.7857835917063,
            "unit": "iter/sec",
            "range": "stddev: 0.000027560989528954565",
            "extra": "mean: 4.019522279621675 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.76598589118504,
            "unit": "iter/sec",
            "range": "stddev: 0.0004353515072038166",
            "extra": "mean: 33.595393199999535 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.16998284639269,
            "unit": "iter/sec",
            "range": "stddev: 0.000922151017612454",
            "extra": "mean: 36.80532320000225 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 124.59603739751448,
            "unit": "iter/sec",
            "range": "stddev: 0.00015242542021044647",
            "extra": "mean: 8.025937428568243 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3263.2111898961593,
            "unit": "iter/sec",
            "range": "stddev: 0.000008570771800332443",
            "extra": "mean: 306.4466078984675 usec\nrounds: 2127"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.43815233707055,
            "unit": "iter/sec",
            "range": "stddev: 0.0003750697087584524",
            "extra": "mean: 40.91962379999927 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.0499907417411,
            "unit": "iter/sec",
            "range": "stddev: 0.00019296468929629211",
            "extra": "mean: 39.920174434782844 msec\nrounds: 23"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9fab2fe1e3665d6bf78686e0777a6f887104bdeb",
          "message": "docs: update for PyPI release (#71)\n\n* docs: update for PyPI release\n\nAdd PyPI and Zenodo badges, update install instructions to use\n`pip install asdex`, remove alpha warnings, and bump development\nstatus to Beta.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add changelog and release process\n\nAdd CHANGELOG.md with v0.1.0 entry, document the release workflow\nin CLAUDE.md, and add changelog badges to README and docs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-26T15:36:42+01:00",
          "tree_id": "158c48d802d9e029e35e7ab3d28f28113e0280ab",
          "url": "https://github.com/adrhill/asdex/commit/9fab2fe1e3665d6bf78686e0777a6f887104bdeb"
        },
        "date": 1772116647877,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 778.5735520380703,
            "unit": "iter/sec",
            "range": "stddev: 0.004139331950011038",
            "extra": "mean: 1.284400166666723 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3303.3279385346764,
            "unit": "iter/sec",
            "range": "stddev: 0.00003415391712093647",
            "extra": "mean: 302.7250150778521 usec\nrounds: 2056"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 49.31472407371305,
            "unit": "iter/sec",
            "range": "stddev: 0.002460357763880446",
            "extra": "mean: 20.277919400000144 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 99.65040445806217,
            "unit": "iter/sec",
            "range": "stddev: 0.00031403890139542005",
            "extra": "mean: 10.035082200001 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.4715495915617,
            "unit": "iter/sec",
            "range": "stddev: 0.014097752702912426",
            "extra": "mean: 46.57325712500038 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.66276839016794,
            "unit": "iter/sec",
            "range": "stddev: 0.000034186939337281794",
            "extra": "mean: 4.021510765258333 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.374431013599803,
            "unit": "iter/sec",
            "range": "stddev: 0.0005841344541746195",
            "extra": "mean: 34.043212599999606 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 25.557556496408914,
            "unit": "iter/sec",
            "range": "stddev: 0.0002647480212587092",
            "extra": "mean: 39.12737120000145 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 117.95496290785773,
            "unit": "iter/sec",
            "range": "stddev: 0.0003489185386163995",
            "extra": "mean: 8.477812000001771 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3310.4264075236433,
            "unit": "iter/sec",
            "range": "stddev: 0.00000798715389942861",
            "extra": "mean: 302.07588899342056 usec\nrounds: 2126"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.519616868076344,
            "unit": "iter/sec",
            "range": "stddev: 0.00045393149461855323",
            "extra": "mean: 42.51769939999832 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 23.92903470594744,
            "unit": "iter/sec",
            "range": "stddev: 0.00030667167925327006",
            "extra": "mean: 41.790235681820256 msec\nrounds: 22"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f491393be9797f1f958b24d224433493357585cd",
          "message": "refactor(decompression): factor out primal in `fwd_over_rev` and `rev_over_rev` (#72)\n\nUse `jax.linearize` for `fwd_over_rev` and `jax.vjp` for `rev_over_rev`\nto compute the grad(f)(x) primal once instead of once per color.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-26T16:58:59+01:00",
          "tree_id": "e2e5927c4646a492da80967bda5c76c4cd955af4",
          "url": "https://github.com/adrhill/asdex/commit/f491393be9797f1f958b24d224433493357585cd"
        },
        "date": 1772121572550,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 858.8058690855697,
            "unit": "iter/sec",
            "range": "stddev: 0.0028810869528661616",
            "extra": "mean: 1.164407505813589 msec\nrounds: 172"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3264.3967186598443,
            "unit": "iter/sec",
            "range": "stddev: 0.000032408445122880096",
            "extra": "mean: 306.3353158897112 usec\nrounds: 2102"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.15769934765121,
            "unit": "iter/sec",
            "range": "stddev: 0.0006069574771650822",
            "extra": "mean: 16.351170999999454 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 106.27039315559401,
            "unit": "iter/sec",
            "range": "stddev: 0.00028756321178241236",
            "extra": "mean: 9.40995860000129 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.394712020594287,
            "unit": "iter/sec",
            "range": "stddev: 0.014355126764983744",
            "extra": "mean: 46.74052163157944 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.83256994519232,
            "unit": "iter/sec",
            "range": "stddev: 0.0004083114931895852",
            "extra": "mean: 4.187033620370461 msec\nrounds: 216"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.806800468117906,
            "unit": "iter/sec",
            "range": "stddev: 0.0014626192580429855",
            "extra": "mean: 32.46036540000006 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 26.90202509513419,
            "unit": "iter/sec",
            "range": "stddev: 0.000843056783558688",
            "extra": "mean: 37.171922800000345 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 102.46578832919414,
            "unit": "iter/sec",
            "range": "stddev: 0.009087232609035397",
            "extra": "mean: 9.759354964286006 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3306.2116268496843,
            "unit": "iter/sec",
            "range": "stddev: 0.000007570836475823397",
            "extra": "mean: 302.46097735517543 usec\nrounds: 2208"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.531318609499646,
            "unit": "iter/sec",
            "range": "stddev: 0.0005596879770074287",
            "extra": "mean: 40.76421719999814 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.224286081178473,
            "unit": "iter/sec",
            "range": "stddev: 0.0002589483007111055",
            "extra": "mean: 39.644333115384654 msec\nrounds: 26"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3fdfe29c0f5ad72158a4945444b83285d2705068",
          "message": "fix(interpret): handle `batch_group_count > 1` in conv handler (#73)\n\n* fix(interpret): handle `batch_group_count > 1` in conv handler\n\nRemove conservative fallback for `batch_group_count > 1`.\nEach output channel group now correctly maps to its corresponding\ninput batch, producing a precise block-diagonal pattern (64/576 nnz)\ninstead of the previous all-ones conservative result.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(conv): add batch_group_count edge case tests\n\nMulti-channel inputs (C_in=2, batch_group_count=3) and degenerate\nbatch_group_count=N with manually constructed expected patterns.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(conv): clean up batch_group_count implementation\n\nHoist `n_out_batches` and `channels_per_batch_group` out of the\nper-element loop, fix imprecise comments with the exact formula\n`in_batch = out_batch + group * n_out_batches`, document mutual\nexclusivity with `feature_group_count`, and add a test confirming\n`batch_group_count=1` is identity.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-26T18:08:52+01:00",
          "tree_id": "fe356bf77e478536a59fdf00e122120bbb18a560",
          "url": "https://github.com/adrhill/asdex/commit/3fdfe29c0f5ad72158a4945444b83285d2705068"
        },
        "date": 1772125766598,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 825.8267720314967,
            "unit": "iter/sec",
            "range": "stddev: 0.0035606939320637202",
            "extra": "mean: 1.2109077034885234 msec\nrounds: 172"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3335.6967744158164,
            "unit": "iter/sec",
            "range": "stddev: 0.00001019626954744871",
            "extra": "mean: 299.78744101376867 usec\nrounds: 2170"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.55016656381888,
            "unit": "iter/sec",
            "range": "stddev: 0.0005932107805229442",
            "extra": "mean: 16.515231200000358 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 104.47971400977589,
            "unit": "iter/sec",
            "range": "stddev: 0.0003419759978114754",
            "extra": "mean: 9.571235999999317 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.064966369231325,
            "unit": "iter/sec",
            "range": "stddev: 0.016487749607768096",
            "extra": "mean: 49.83810994736839 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.78721120016797,
            "unit": "iter/sec",
            "range": "stddev: 0.00003084018833405464",
            "extra": "mean: 4.003407521126636 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.807492917096106,
            "unit": "iter/sec",
            "range": "stddev: 0.0013430837025436772",
            "extra": "mean: 32.459635799999376 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 28.540947491870675,
            "unit": "iter/sec",
            "range": "stddev: 0.0006169419905952192",
            "extra": "mean: 35.03737920000134 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 102.34690502851964,
            "unit": "iter/sec",
            "range": "stddev: 0.008949646711453584",
            "extra": "mean: 9.770691157894257 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3333.3267564558914,
            "unit": "iter/sec",
            "range": "stddev: 0.000012769840725856114",
            "extra": "mean: 300.00059192013765 usec\nrounds: 2203"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.85963689666262,
            "unit": "iter/sec",
            "range": "stddev: 0.0006777380128637502",
            "extra": "mean: 40.225849000000835 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.552205820801074,
            "unit": "iter/sec",
            "range": "stddev: 0.0001643212977511268",
            "extra": "mean: 39.135564538461026 msec\nrounds: 26"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "adc9d0ab80a2b317f48de69892cfe6409fd28ca3",
          "message": "test: suppress expected warnings for clean pytest output (#74)\n\nReduces token usage in CI and agentic coding contexts\nby eliminating 16 warnings from the test output.\n\n- Add `filterwarnings` in `pyproject.toml` for JAX's `jax_pmap_shmap_merge` deprecation\n- Add `@pytest.mark.filterwarnings` for `DenseColoringWarning` on 12 tests that intentionally trigger dense colorings\n- Add `@pytest.mark.filterwarnings` for int64 truncation warning on `test_gather_indices_through_convert_element_type`\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-02T14:01:05+01:00",
          "tree_id": "46b9596850aa6cf5a4784ae2100bb7b4bd934f75",
          "url": "https://github.com/adrhill/asdex/commit/adc9d0ab80a2b317f48de69892cfe6409fd28ca3"
        },
        "date": 1772456500852,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 751.0118912940974,
            "unit": "iter/sec",
            "range": "stddev: 0.0037318159618115096",
            "extra": "mean: 1.3315368392860754 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3357.298089351733,
            "unit": "iter/sec",
            "range": "stddev: 0.000007715421877693924",
            "extra": "mean: 297.8585676296298 usec\nrounds: 2144"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 70.2428204390581,
            "unit": "iter/sec",
            "range": "stddev: 0.000596983944104034",
            "extra": "mean: 14.236330400024144 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 126.10870358388918,
            "unit": "iter/sec",
            "range": "stddev: 0.0003005204706103368",
            "extra": "mean: 7.929666799998357 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.65856918094952,
            "unit": "iter/sec",
            "range": "stddev: 0.010512533163302683",
            "extra": "mean: 44.13341336842941 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.86861096043262,
            "unit": "iter/sec",
            "range": "stddev: 0.00012763944207684163",
            "extra": "mean: 4.002103330051139 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.34229093593426,
            "unit": "iter/sec",
            "range": "stddev: 0.0008962287037419359",
            "extra": "mean: 28.294713599996157 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.17469569831412,
            "unit": "iter/sec",
            "range": "stddev: 0.0006732340394088092",
            "extra": "mean: 30.143456599989804 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 100.8939136544238,
            "unit": "iter/sec",
            "range": "stddev: 0.011189787043090156",
            "extra": "mean: 9.911400636366867 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3323.8868067054623,
            "unit": "iter/sec",
            "range": "stddev: 0.000009292258850933613",
            "extra": "mean: 300.85260363940324 usec\nrounds: 2253"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.753102802449924,
            "unit": "iter/sec",
            "range": "stddev: 0.0003131853619162078",
            "extra": "mean: 36.03200719999222 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 27.96735790857394,
            "unit": "iter/sec",
            "range": "stddev: 0.00026646939634642906",
            "extra": "mean: 35.75596962963135 msec\nrounds: 27"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "91abf2229f40b11778b0c40134f7d4dc15a47efa",
          "message": "test(cutest): add CUTEst integration tests and fix discovered bugs (#75)\n\n* test(cutest): add CUTEst integration tests via sif2jax\n\nCompare asdex sparsity detection against ground-truth patterns from\nCUTEst benchmark problems. Tests assert no false negatives and warn\non conservativeness (extra nonzeros).\n\n- Add `sif2jax` test dependency and `cutest` pytest marker\n- Add pycutest-based fixture generation script (`tests/setup/`)\n- Add parametrized tests for Hessian, eq/ineq Jacobian sparsity\n- Fixtures discovered via glob, not manifest files\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): fix CUTEst-discovered bugs in `dot_general`, `gather`, and `erf`\n\n- `dot_general`: broadcast flat indices to output size for scalar\n  operands, and disable zero-skipping when constant size mismatches\n  operand shape (fixes HILBERTA)\n- `gather`: bail out of `_try_single_dim_gather` when reshape produces\n  wrong ndim for the permutation (fixes POWERSUM)\n- Add `erf` to unary elementwise dispatch (fixes HS68, HS69)\n- Update TODO.md: remove fixed bugs, keep `cumsum` for follow-up\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(cutest): catalog conservative patterns and handler improvements\n\nDocument ~60 conservative CUTEst patterns in TODO.md with root cause\nanalysis mapping each to specific handler limitations: VJP index\nexpressions, gather/scatter composition, structural zeros, and\n`reduce_sum` lattice limitations. Add prioritized summary of handler\nimprovements, with const propagation through comparison+select chains\nas the highest-impact fix (~12–15 problems).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(cutest): add nnz baseline regression and improvement detection\n\nTrack (detected_nnz, target_nnz) for each CUTEst problem in\n`EXPECTED_NNZ`. Tests fail on regressions (nnz increases) and\nimprovements (nnz decreases), prompting developers to update\nbaselines when handlers improve.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix weird import\n\n* refactor(cutest): improve CUTEst test quality and dot_general scalar handling\n\n- Broadcast scalar constants instead of discarding them in `dot_general`,\n  preserving zero-skipping for cases like `jnp.dot(0.0, x)`\n- Narrow xfail to `NotImplementedError` only; `IndexError`/`ValueError`\n  are real bugs that should fail\n- Replace `_find_problem` linear scan with dict lookup\n- Extract shared `_run_sparsity_test` helper and named extractors\n  (`_objective`, `_eq_constraint`, `_ineq_constraint`)\n- Split `EXPECTED_NNZ` into three per-kind dicts\n- Improve comments in `_gather.py` and `_dot_general.py`\n- Document CUTEst tests in `tests/CLAUDE.md` and sif2jax problem\n  structure in the test module docstring\n- Add `test_scalar_zero_times_vector` for the broadcast fix\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(gather): squeeze size-1 axis for scalar-index gathers\n\nWhen `start_indices` is a 1D index vector with no batch dims (e.g. shape\n`(1,)`), `np.take` leaves a size-1 axis that doesn't match the scalar\noutput shape, causing `transpose(perm)` to fail with \"axes don't match\narray\". Squeeze the axis early so the permutation logic sees the correct\nndim.\n\nFixes the POWERSUM CUTEst failure.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-03T16:42:44+01:00",
          "tree_id": "a7e0bab9f73c421fb0dfb042666174c71a4ca32a",
          "url": "https://github.com/adrhill/asdex/commit/91abf2229f40b11778b0c40134f7d4dc15a47efa"
        },
        "date": 1772552602083,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 801.9361873141509,
            "unit": "iter/sec",
            "range": "stddev: 0.0032810630364491083",
            "extra": "mean: 1.2469820115603032 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3342.2844776173315,
            "unit": "iter/sec",
            "range": "stddev: 0.0000162292458179323",
            "extra": "mean: 299.19655454130765 usec\nrounds: 2191"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 70.75870683764339,
            "unit": "iter/sec",
            "range": "stddev: 0.0005048396192524946",
            "extra": "mean: 14.132536400001072 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 123.73411646801036,
            "unit": "iter/sec",
            "range": "stddev: 0.00035488856650315605",
            "extra": "mean: 8.081845399999565 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.75506028879116,
            "unit": "iter/sec",
            "range": "stddev: 0.015354722654999483",
            "extra": "mean: 48.181021210526346 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 251.95894011084138,
            "unit": "iter/sec",
            "range": "stddev: 0.00005440990886333517",
            "extra": "mean: 3.968900645319755 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 25.29816685081133,
            "unit": "iter/sec",
            "range": "stddev: 0.023418979338653844",
            "extra": "mean: 39.528555799999765 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 34.383475032333884,
            "unit": "iter/sec",
            "range": "stddev: 0.0006671486109850144",
            "extra": "mean: 29.083738599999265 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 104.14907128419831,
            "unit": "iter/sec",
            "range": "stddev: 0.009795867225764436",
            "extra": "mean: 9.6016218644066 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3298.9764356851165,
            "unit": "iter/sec",
            "range": "stddev: 0.000007859857930839927",
            "extra": "mean: 303.1243234061854 usec\nrounds: 2149"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.758652066502716,
            "unit": "iter/sec",
            "range": "stddev: 0.0007802578286081197",
            "extra": "mean: 36.02480400000161 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.129354270208847,
            "unit": "iter/sec",
            "range": "stddev: 0.00134111487476326",
            "extra": "mean: 35.55005174999972 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dcf373d738aa38f86750b970a2490e5b2f065d81",
          "message": "feat(interpret): add `cumsum` primitive handler (#76)\n\nLower/upper-triangular dependency propagation along the scan axis.\nResolves HADAMARD, HS91, HS92 CUTEst problems (previously xfailed).\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-03T17:32:40+01:00",
          "tree_id": "32996d43c7d9c1173956b74eb9ef4ffdc2226839",
          "url": "https://github.com/adrhill/asdex/commit/dcf373d738aa38f86750b970a2490e5b2f065d81"
        },
        "date": 1772555595789,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 796.8431309981796,
            "unit": "iter/sec",
            "range": "stddev: 0.0036595499856706976",
            "extra": "mean: 1.254952149424106 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3376.806779249813,
            "unit": "iter/sec",
            "range": "stddev: 0.000012475507258556343",
            "extra": "mean: 296.1377613148948 usec\nrounds: 2099"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.83978028069592,
            "unit": "iter/sec",
            "range": "stddev: 0.0004408236998953378",
            "extra": "mean: 14.318487200000618 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 123.48818966354033,
            "unit": "iter/sec",
            "range": "stddev: 0.00020844929987866228",
            "extra": "mean: 8.09794040000611 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.72168146001403,
            "unit": "iter/sec",
            "range": "stddev: 0.01750193012707675",
            "extra": "mean: 48.25863199999808 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.60043329462624,
            "unit": "iter/sec",
            "range": "stddev: 0.0006182807918804141",
            "extra": "mean: 4.071654054455124 msec\nrounds: 202"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.30779439121641,
            "unit": "iter/sec",
            "range": "stddev: 0.0010648412617200388",
            "extra": "mean: 28.322358200000508 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.90177179948186,
            "unit": "iter/sec",
            "range": "stddev: 0.0009623672199245835",
            "extra": "mean: 29.49698340000282 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 103.1799628710597,
            "unit": "iter/sec",
            "range": "stddev: 0.010620499849701957",
            "extra": "mean: 9.691804224136659 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3298.6261083132053,
            "unit": "iter/sec",
            "range": "stddev: 0.000033129481221850035",
            "extra": "mean: 303.1565164296122 usec\nrounds: 2252"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.94349840271579,
            "unit": "iter/sec",
            "range": "stddev: 0.0006511872098735962",
            "extra": "mean: 35.78649979999682 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.52138840663909,
            "unit": "iter/sec",
            "range": "stddev: 0.00046548097843832406",
            "extra": "mean: 35.06140675000324 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "f0e922ca0a5d5bfd7f4ace21f2e97e3a14d66732",
          "message": "asdex `v0.1.1`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-03T17:52:05+01:00",
          "tree_id": "12f41721ddb060d5373d7c8317fdb3ae73207f6e",
          "url": "https://github.com/adrhill/asdex/commit/f0e922ca0a5d5bfd7f4ace21f2e97e3a14d66732"
        },
        "date": 1772556774436,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 832.774955399808,
            "unit": "iter/sec",
            "range": "stddev: 0.0031374143227404206",
            "extra": "mean: 1.2008046033515845 msec\nrounds: 179"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3253.8364841129314,
            "unit": "iter/sec",
            "range": "stddev: 0.000033963846848927795",
            "extra": "mean: 307.32951851839056 usec\nrounds: 2160"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 70.92756444461985,
            "unit": "iter/sec",
            "range": "stddev: 0.0004022723879300175",
            "extra": "mean: 14.098890999997593 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 125.45891931752107,
            "unit": "iter/sec",
            "range": "stddev: 0.0002142184524442176",
            "extra": "mean: 7.970736599995121 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.425434877250552,
            "unit": "iter/sec",
            "range": "stddev: 0.009612514500091917",
            "extra": "mean: 44.592223315787216 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.9738753356507,
            "unit": "iter/sec",
            "range": "stddev: 0.0000285258396332189",
            "extra": "mean: 4.016485659998921 msec\nrounds: 200"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.32543318642599,
            "unit": "iter/sec",
            "range": "stddev: 0.0005732119211885219",
            "extra": "mean: 28.308216199999947 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.96775382704674,
            "unit": "iter/sec",
            "range": "stddev: 0.000465107144769527",
            "extra": "mean: 29.439685799999893 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 113.91362304017626,
            "unit": "iter/sec",
            "range": "stddev: 0.007561175355791115",
            "extra": "mean: 8.778581290907669 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3285.873387986475,
            "unit": "iter/sec",
            "range": "stddev: 0.000008343934696381617",
            "extra": "mean: 304.3330895390289 usec\nrounds: 2256"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.12466536921918,
            "unit": "iter/sec",
            "range": "stddev: 0.0004067665254012527",
            "extra": "mean: 35.5559785999958 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.649579391073875,
            "unit": "iter/sec",
            "range": "stddev: 0.00022068811122327527",
            "extra": "mean: 34.90452639285735 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "250e637d6558018109857e448bd4c78d1a4f289d",
          "message": "claude: rename `/prepare-release` to `/make-release` and improve skill\n\n- Add badge-enhancement mapping and skip rule for internal commits\n- Fix changelog heading format to use backtick-wrapped version\n- Add explicit user confirmation checkpoint before committing\n- Move release instructions from CLAUDE.md into the skill\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-03T17:59:47+01:00",
          "tree_id": "a6073965a2ab405ac376040ffb5b7fd2d1367d06",
          "url": "https://github.com/adrhill/asdex/commit/250e637d6558018109857e448bd4c78d1a4f289d"
        },
        "date": 1772557249966,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 812.729607996265,
            "unit": "iter/sec",
            "range": "stddev: 0.003354627032392647",
            "extra": "mean: 1.230421520467845 msec\nrounds: 171"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3326.538798112058,
            "unit": "iter/sec",
            "range": "stddev: 0.000009906268953906871",
            "extra": "mean: 300.6127571900077 usec\nrounds: 2121"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.46355078000352,
            "unit": "iter/sec",
            "range": "stddev: 0.0005382047069699763",
            "extra": "mean: 14.396039199998256 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 126.31614156332758,
            "unit": "iter/sec",
            "range": "stddev: 0.00028184683731470475",
            "extra": "mean: 7.916644599998791 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.699118163095427,
            "unit": "iter/sec",
            "range": "stddev: 0.01696141093135015",
            "extra": "mean: 48.311236842104 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.10905862828267,
            "unit": "iter/sec",
            "range": "stddev: 0.000047704068834494334",
            "extra": "mean: 4.030485648241491 msec\nrounds: 199"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 25.010547448071364,
            "unit": "iter/sec",
            "range": "stddev: 0.02519765299740602",
            "extra": "mean: 39.9831311999975 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.58418931595609,
            "unit": "iter/sec",
            "range": "stddev: 0.0010537741178331208",
            "extra": "mean: 29.775915999999825 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 102.95057446936218,
            "unit": "iter/sec",
            "range": "stddev: 0.010790777411007547",
            "extra": "mean: 9.713398931033623 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3265.4350738132725,
            "unit": "iter/sec",
            "range": "stddev: 0.000009412939630146652",
            "extra": "mean: 306.2379062500334 usec\nrounds: 2240"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.842053633440816,
            "unit": "iter/sec",
            "range": "stddev: 0.0009647752620261265",
            "extra": "mean: 35.91689079999867 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.485989815489557,
            "unit": "iter/sec",
            "range": "stddev: 0.00017355816638819037",
            "extra": "mean: 35.104976392859605 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "760e58333525f0c8524f9f6b30f2978024d21fc7",
          "message": "claude: rename `/prepare-release` to `/make-release` and improve skill\n\n- Add badge-enhancement mapping and skip rule for internal commits\n- Fix changelog heading format to use backtick-wrapped version\n- Add explicit user confirmation checkpoint before committing\n- Move release instructions from CLAUDE.md into the skill\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-03T18:01:19+01:00",
          "tree_id": "6665c9b312dc982393652f3577127df5476693b0",
          "url": "https://github.com/adrhill/asdex/commit/760e58333525f0c8524f9f6b30f2978024d21fc7"
        },
        "date": 1772557355975,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 816.9758606964408,
            "unit": "iter/sec",
            "range": "stddev: 0.0032987053432372232",
            "extra": "mean: 1.2240263734935057 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3327.006946476331,
            "unit": "iter/sec",
            "range": "stddev: 0.00001907495498967402",
            "extra": "mean: 300.57045749757475 usec\nrounds: 2094"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.30936914764101,
            "unit": "iter/sec",
            "range": "stddev: 0.0005873568033416855",
            "extra": "mean: 14.428063800001212 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 126.32688709124167,
            "unit": "iter/sec",
            "range": "stddev: 0.000270350453638766",
            "extra": "mean: 7.915971200000627 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.04388177856436,
            "unit": "iter/sec",
            "range": "stddev: 0.012280045865774504",
            "extra": "mean: 45.36406110526358 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.94510580615116,
            "unit": "iter/sec",
            "range": "stddev: 0.00003756921050075017",
            "extra": "mean: 4.000878499999779 msec\nrounds: 196"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.89913188288522,
            "unit": "iter/sec",
            "range": "stddev: 0.0010064485758516783",
            "extra": "mean: 27.85582679999976 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.97721422777607,
            "unit": "iter/sec",
            "range": "stddev: 0.0007054018238774159",
            "extra": "mean: 29.43148880000024 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 104.47226650763106,
            "unit": "iter/sec",
            "range": "stddev: 0.009887397306796082",
            "extra": "mean: 9.571918303571561 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3268.651441050764,
            "unit": "iter/sec",
            "range": "stddev: 0.000010924173121697702",
            "extra": "mean: 305.9365668180676 usec\nrounds: 2200"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.173206098852003,
            "unit": "iter/sec",
            "range": "stddev: 0.000653742185960293",
            "extra": "mean: 35.49471780000033 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.651504879004907,
            "unit": "iter/sec",
            "range": "stddev: 0.00042911491591416783",
            "extra": "mean: 34.902180678571426 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7f1bcee4edd992d4a83dae132abfd30c1cfccb30",
          "message": "fix(gather): handle all `GatherDimensionNumbers` configurations (#77)\n\n* fix(gather): add general handler for any `GatherDimensionNumbers` config\n\nAdd `_try_general_gather` that simulates XLA gather semantics on a\nposition map, handling mismatched `start_index_map`, partial slices,\nand `operand_batching_dims`. Runs as a third fallback after the two\nexisting specialized pattern matchers.\n\nResolves all 5 gather fallback cases (TODO §5), reducing fallback\ntest count from 22 to 17.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(gather): remove specialized handlers, inline general gather\n\nThe general handler covers all `GatherDimensionNumbers` configurations,\nmaking `_try_single_dim_gather` and `_try_multi_dim_gather` redundant.\nInlines the remaining logic directly into `prop_gather`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-03T19:24:18+01:00",
          "tree_id": "3c52fd65ba7898328f2d8e76158ffa085a608663",
          "url": "https://github.com/adrhill/asdex/commit/7f1bcee4edd992d4a83dae132abfd30c1cfccb30"
        },
        "date": 1772562292660,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 849.7865816528011,
            "unit": "iter/sec",
            "range": "stddev: 0.002800711807866484",
            "extra": "mean: 1.1767660511360862 msec\nrounds: 176"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3322.6527243099767,
            "unit": "iter/sec",
            "range": "stddev: 0.000008350245290242278",
            "extra": "mean: 300.96434474887 usec\nrounds: 2190"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 70.19533860553557,
            "unit": "iter/sec",
            "range": "stddev: 0.00038527381304743693",
            "extra": "mean: 14.24596020000024 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 125.7245347783584,
            "unit": "iter/sec",
            "range": "stddev: 0.00023738397098243882",
            "extra": "mean: 7.953897000000154 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.339429491700592,
            "unit": "iter/sec",
            "range": "stddev: 0.0141927226821178",
            "extra": "mean: 46.86160894736776 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.36025178920795,
            "unit": "iter/sec",
            "range": "stddev: 0.0000427327225102587",
            "extra": "mean: 4.026409189054676 msec\nrounds: 201"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.71646145939963,
            "unit": "iter/sec",
            "range": "stddev: 0.0009197203804671878",
            "extra": "mean: 28.804778999999314 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.648530106695304,
            "unit": "iter/sec",
            "range": "stddev: 0.0008139639598613483",
            "extra": "mean: 29.718980200000544 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 105.2769193650081,
            "unit": "iter/sec",
            "range": "stddev: 0.009380892233353186",
            "extra": "mean: 9.498758189654811 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3308.7631035375266,
            "unit": "iter/sec",
            "range": "stddev: 0.00000806063863391197",
            "extra": "mean: 302.22774151792896 usec\nrounds: 2240"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.46524296767296,
            "unit": "iter/sec",
            "range": "stddev: 0.0004615316424262838",
            "extra": "mean: 35.13056259999843 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 29.00095424739778,
            "unit": "iter/sec",
            "range": "stddev: 0.00017832658292664893",
            "extra": "mean: 34.48162400000093 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0292518b956c13e68c3eb0373d32d31bddf3c7d5",
          "message": "feat(interpret): track value bounds for dynamic-index sparsity (#78)\n\n* feat(interpret): track value bounds for tighter dynamic-index sparsity\n\nWhen `gather`, `scatter`, `dynamic_slice`, and `dynamic_update_slice`\nhave indices from bounded ops like `argmax`, enumerate all possible\nindex values and union the resulting patterns instead of falling back\nto fully conservative all-ones sparsity.\n\nFor example, `argmax(x[:2])` can only return 0 or 1, so\n`x[jnp.array([0, 1]) + jnp.argmax(x[:2])]` gathers from either\n`x[[0,1]]` or `x[[1,2]]` — yielding a 2-nonzero-per-row pattern\ninstead of a dense row.\n\nKey changes:\n- Add `ValueBounds` type and `atom_value_bounds` helper to `_commons`\n- Source bounds from `argmax`/`argmin` in `prop_dispatch`\n- Propagate bounds through `add`/`sub`, `convert_element_type`,\n  `broadcast_in_dim`, comparisons, and `select_n`\n- Forward bounds into/out of nested jaxprs (`jit`, `custom_jvp`/`vjp`)\n- Add bounded enumeration (capped at 64 combinations) to\n  `_dynamic_slice`, `_gather`, and `_scatter` handlers\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): move bounds propagation into handler files\n\nMove value bounds helpers out of `__init__.py` into their\nrespective handler modules, following the one-handler-per-file pattern:\n\n- `_propagate_const_comparison_from_bounds` → `_elementwise.py`\n- `_propagate_bounds_binary` → `_elementwise.py`\n- `_propagate_bounds_convert` → `_elementwise.py`\n- `_propagate_bounds_select_n` → `_select.py`\n- `_propagate_bounds_broadcast` → `_broadcast.py`\n- argmax/argmin bounds sourcing → new `_argmax.py`\n\nDeduplicate `_MAX_ENUM_COMBINATIONS` into `_commons.py`\nwith comprehensive documentation.\nMove `_clamp_starts` to `_commons.py` as `clamp_starts`.\n\nEach dispatch case is now a single function call\nwith no `if name in (...)` routing inside helpers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): one handler call per dispatch case\n\nEach dispatch case now calls exactly one `prop_*` function\ninstead of composing 2-3 calls inline.\n\n- Deduplicate `prop_nested_jaxpr` / `prop_custom_call` into\n  `_prop_closed_jaxpr` parameterized by jaxpr key\n- Create composite handlers: `prop_zero_derivative_const`,\n  `prop_comparison`, `prop_binary_const`, `prop_add`, `prop_sub`\n- Make `prop_mul` handle const propagation internally\n- Use `atom_value_bounds` in broadcast and convert instead of\n  manual isinstance checks\n- Eliminate dispatch-within-dispatch in comparison and arithmetic\n  bounds (pass ops as parameters instead of matching on name)\n- Rename \"combo\" → \"candidate_values\" in gather/scatter/dynamic_slice\n- Rename \"Pattern N\" comments in scatter to describe JAX operations\n- Remove `noqa: E712` by using `not np.any()` / `np.all()`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* claude: update `_interpret/CLAUDE.md`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: update TODO.md for resolved items and corrected root cause\n\nRemove dynamic-index fallbacks (§1) and value range tracking (§3),\nboth resolved by `ValueBounds`. Remove POWERSUM xfail (fixed by\ngather handler). Correct CUTEst root cause: const propagation\nthrough `iota → lt → select_n` already works; the real blocker is\nan unhandled `scatter-add` `ScatterDimensionNumbers` configuration.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract bounded enumeration to `enumerate_bounded_patterns`\n\nThe four dynamic-index handlers (gather, scatter, dynamic_slice,\ndynamic_update_slice) shared a ~25-line enumeration block.\nExtract it to a shared helper in `_commons.py` that takes ranges\nand a `make_pattern` callback, simplifying each call site to ~8 lines.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract comparison handlers to `_comparison.py`\n\nMove lt/le/gt/ge from `_elementwise.py` to dedicated `_comparison.py`\nwith individual `prop_lt`, `prop_le`, `prop_gt`, `prop_ge` handlers.\nInline bounds resolution with direct comparisons instead of the\n`swap_bounds`/`true_op`/`false_op` indirection.\n\nAlso rename `_prop_closed_jaxpr` → `prop_closed_jaxpr` to match\nthe `prop_*` naming convention, and condense `CLAUDE.md`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-04T00:55:23+01:00",
          "tree_id": "6b753bd8f534ddc184a767b4c0c3220cadc2734f",
          "url": "https://github.com/adrhill/asdex/commit/0292518b956c13e68c3eb0373d32d31bddf3c7d5"
        },
        "date": 1772582155268,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 781.01343541136,
            "unit": "iter/sec",
            "range": "stddev: 0.0038071527622619014",
            "extra": "mean: 1.2803877048200838 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3225.2975103863187,
            "unit": "iter/sec",
            "range": "stddev: 0.000011920434162580495",
            "extra": "mean: 310.04891696959214 usec\nrounds: 1927"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.7694711301706,
            "unit": "iter/sec",
            "range": "stddev: 0.0009719782967636077",
            "extra": "mean: 14.541336199999932 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 123.55745131114031,
            "unit": "iter/sec",
            "range": "stddev: 0.00025422767857888497",
            "extra": "mean: 8.093401000007816 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.669025182272186,
            "unit": "iter/sec",
            "range": "stddev: 0.010766530050976475",
            "extra": "mean: 44.11305699999963 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.73320580735935,
            "unit": "iter/sec",
            "range": "stddev: 0.000035537277545454584",
            "extra": "mean: 4.171303664973148 msec\nrounds: 197"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.06999627570783,
            "unit": "iter/sec",
            "range": "stddev: 0.0007935772434681811",
            "extra": "mean: 28.51440279999906 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.136198357526624,
            "unit": "iter/sec",
            "range": "stddev: 0.0003884306246078974",
            "extra": "mean: 30.178476999998338 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 114.63950888910735,
            "unit": "iter/sec",
            "range": "stddev: 0.007390725455318031",
            "extra": "mean: 8.7229961964275 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3177.1507792373395,
            "unit": "iter/sec",
            "range": "stddev: 0.000007455997427089697",
            "extra": "mean: 314.74741662718486 usec\nrounds: 2105"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 20.629799729206205,
            "unit": "iter/sec",
            "range": "stddev: 0.027845891001831234",
            "extra": "mean: 48.47356799999716 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.470888625178933,
            "unit": "iter/sec",
            "range": "stddev: 0.00027204663257262315",
            "extra": "mean: 35.12359635714444 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6852cb16b0f354258c8e2dba6c285e730f188119",
          "message": "refactor(interpret): rename interpreter state types and variables (#79)\n\nRename `Deps` → `StateIndices`, `ConstVals` → `StateConsts`,\n`ValueBounds` → `StateBounds` and their corresponding variables\nto make their role as interpreter state explicit.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-04T01:18:49+01:00",
          "tree_id": "9e6a456b91d33ca048c96ecd35d01addeecba9c7",
          "url": "https://github.com/adrhill/asdex/commit/6852cb16b0f354258c8e2dba6c285e730f188119"
        },
        "date": 1772583564773,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 759.666775172282,
            "unit": "iter/sec",
            "range": "stddev: 0.0039599924971865826",
            "extra": "mean: 1.3163666395350957 msec\nrounds: 172"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3252.1233654075872,
            "unit": "iter/sec",
            "range": "stddev: 0.000007445803712412712",
            "extra": "mean: 307.4914102696318 usec\nrounds: 1928"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.99674183585257,
            "unit": "iter/sec",
            "range": "stddev: 0.0005000367917095741",
            "extra": "mean: 14.493438000000936 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 122.6247945176525,
            "unit": "iter/sec",
            "range": "stddev: 0.00019868022816432504",
            "extra": "mean: 8.154957599998625 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.88160121401838,
            "unit": "iter/sec",
            "range": "stddev: 0.01597379810779746",
            "extra": "mean: 50.29775968421029 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.66680349666382,
            "unit": "iter/sec",
            "range": "stddev: 0.000029013239214866816",
            "extra": "mean: 4.070554042168665 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 25.377205127496072,
            "unit": "iter/sec",
            "range": "stddev: 0.02263228365581491",
            "extra": "mean: 39.405442600001095 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.44937322296814,
            "unit": "iter/sec",
            "range": "stddev: 0.0005168638393714832",
            "extra": "mean: 29.895926400000405 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 102.06528452122916,
            "unit": "iter/sec",
            "range": "stddev: 0.010126829305320441",
            "extra": "mean: 9.797650637930706 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3252.4832052625766,
            "unit": "iter/sec",
            "range": "stddev: 0.0000090875141618819",
            "extra": "mean: 307.457390827409 usec\nrounds: 2006"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.948841456800555,
            "unit": "iter/sec",
            "range": "stddev: 0.0004332723970012719",
            "extra": "mean: 35.77965839999706 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.226089480757867,
            "unit": "iter/sec",
            "range": "stddev: 0.0003616917222293434",
            "extra": "mean: 35.42821617857176 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "13a979c96b03e85c13849ef1bd700f854294a7fa",
          "message": "feat(interpret): zero-skipping and bounds propagation for `div`, `mul`, `integer_pow` (#80)\n\n* feat(interpret): zero-skipping and bounds propagation for `div`, `mul`, `integer_pow`\n\nAdd `prop_div` handler with zero-skipping for zero numerators (`d(0/y)/dy = 0`)\nand interval arithmetic bounds propagation.\nExtend `prop_mul` and `prop_integer_pow` with bounds propagation.\nAdd const propagation and zero-skipping to `prop_integer_pow`.\n\nUpdate TODO.md to reflect completed work and add new `select_n` bounds\nmerging item.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract `clear_where_zero` helper for zero-skipping\n\nDeduplicate the broadcast-and-clear pattern shared by `prop_mul`,\n`prop_div`, and `prop_integer_pow` into `clear_where_zero` and\n`broadcast_to_output` in `_commons.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-04T18:11:56+01:00",
          "tree_id": "38f5ac7df5dc370c8fc7b0e60ee2403202e90485",
          "url": "https://github.com/adrhill/asdex/commit/13a979c96b03e85c13849ef1bd700f854294a7fa"
        },
        "date": 1772644358324,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 710.4410691772279,
            "unit": "iter/sec",
            "range": "stddev: 0.004454468939251355",
            "extra": "mean: 1.4075762837839803 msec\nrounds: 148"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3265.4805031028404,
            "unit": "iter/sec",
            "range": "stddev: 0.000007560463648700156",
            "extra": "mean: 306.23364587533314 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.25267704878667,
            "unit": "iter/sec",
            "range": "stddev: 0.0006293221690130605",
            "extra": "mean: 14.43987500000219 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 122.96658761720158,
            "unit": "iter/sec",
            "range": "stddev: 0.00035917971855256163",
            "extra": "mean: 8.132290399998965 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.334018357872623,
            "unit": "iter/sec",
            "range": "stddev: 0.01914850448659569",
            "extra": "mean: 49.17867105263209 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.4616101503077,
            "unit": "iter/sec",
            "range": "stddev: 0.0003174348289008534",
            "extra": "mean: 4.141445090909105 msec\nrounds: 198"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.870941460043795,
            "unit": "iter/sec",
            "range": "stddev: 0.001333139268558142",
            "extra": "mean: 28.677172400000472 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.35022766932968,
            "unit": "iter/sec",
            "range": "stddev: 0.0008207268618642114",
            "extra": "mean: 29.984802800001376 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.16971977938078,
            "unit": "iter/sec",
            "range": "stddev: 0.012846696036785846",
            "extra": "mean: 10.507543810343943 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3001.811342140349,
            "unit": "iter/sec",
            "range": "stddev: 0.00006597634446158058",
            "extra": "mean: 333.13219453924137 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.216554288974557,
            "unit": "iter/sec",
            "range": "stddev: 0.0006881288563684877",
            "extra": "mean: 35.44018839999694 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.908256992038694,
            "unit": "iter/sec",
            "range": "stddev: 0.0004900538386516147",
            "extra": "mean: 34.592192821428114 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "781c4d8b6d92f2316df664666cd33d3d6559ffa4",
          "message": "feat(interpret): handle all `ScatterDimensionNumbers` configurations (#81)\n\n* feat(scatter): handle all `ScatterDimensionNumbers` configurations\n\nReplace the 3-pattern approach in `_scatter_for_indices` with a generic\n`_scatter_flat_map` that simulates XLA scatter semantics for any\n`ScatterDimensionNumbers` configuration, mirroring `_gather_flat_map`.\n\nThis removes the conservative fallback for unrecognized scatter patterns\n(e.g. partial-row scatter `mat.at[0, :2].set(...)`) and enables precise\nsparsity detection for all static-index scatter operations.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(cutest): update Hessian baselines for scatter improvements\n\n5 problems now match ground truth exactly:\n`COATING`, `ERRINROS`, `GENROSE`, `LUKSAN17LS`, `LUKSAN21LS`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-04T20:41:04+01:00",
          "tree_id": "2558b55aab20f910b6bcc76a169b2f87555b23ff",
          "url": "https://github.com/adrhill/asdex/commit/781c4d8b6d92f2316df664666cd33d3d6559ffa4"
        },
        "date": 1772653311836,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 755.532446694226,
            "unit": "iter/sec",
            "range": "stddev: 0.004171938313368813",
            "extra": "mean: 1.3235698934909055 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3431.246000319195,
            "unit": "iter/sec",
            "range": "stddev: 0.000015682243640215567",
            "extra": "mean: 291.43931968357094 usec\nrounds: 2149"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.67353878147891,
            "unit": "iter/sec",
            "range": "stddev: 0.0004527310885413875",
            "extra": "mean: 14.352651199996558 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 121.81548734820473,
            "unit": "iter/sec",
            "range": "stddev: 0.0003236328607859488",
            "extra": "mean: 8.209136799999328 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.30231070545105,
            "unit": "iter/sec",
            "range": "stddev: 0.0009366264505560674",
            "extra": "mean: 42.91419905263183 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 254.82740600644985,
            "unit": "iter/sec",
            "range": "stddev: 0.000278508015165429",
            "extra": "mean: 3.924224696517491 msec\nrounds: 201"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 33.38087348318799,
            "unit": "iter/sec",
            "range": "stddev: 0.001133212659225227",
            "extra": "mean: 29.95727480000312 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.98021976068203,
            "unit": "iter/sec",
            "range": "stddev: 0.0006408155077328681",
            "extra": "mean: 29.428885600000854 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.66924206846892,
            "unit": "iter/sec",
            "range": "stddev: 0.011741752425996864",
            "extra": "mean: 10.79106699999928 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3392.750507171826,
            "unit": "iter/sec",
            "range": "stddev: 0.00000805381090839917",
            "extra": "mean: 294.7461058177229 usec\nrounds: 2183"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.223459020779426,
            "unit": "iter/sec",
            "range": "stddev: 0.0008334055237318831",
            "extra": "mean: 38.13379460000306 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 27.583228556056657,
            "unit": "iter/sec",
            "range": "stddev: 0.0011052321498506855",
            "extra": "mean: 36.25391414814719 msec\nrounds: 27"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "1a3b82b8ba055fb255bc963d7bbb6848b9359129",
          "message": "asdex `v0.1.2`",
          "timestamp": "2026-03-04T20:44:13+01:00",
          "tree_id": "e6a507be07a82549f9351358df7f21069563d412",
          "url": "https://github.com/adrhill/asdex/commit/1a3b82b8ba055fb255bc963d7bbb6848b9359129"
        },
        "date": 1772653495442,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 760.1971154234769,
            "unit": "iter/sec",
            "range": "stddev: 0.004377200691886146",
            "extra": "mean: 1.3154482958580263 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3377.493913830086,
            "unit": "iter/sec",
            "range": "stddev: 0.000008624713404194911",
            "extra": "mean: 296.07751353902444 usec\nrounds: 2068"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.76315344569538,
            "unit": "iter/sec",
            "range": "stddev: 0.0005282731686898068",
            "extra": "mean: 14.54267220001384 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 123.00848040106506,
            "unit": "iter/sec",
            "range": "stddev: 0.00039536581364151046",
            "extra": "mean: 8.129520800025603 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.829835098043556,
            "unit": "iter/sec",
            "range": "stddev: 0.01263510753677516",
            "extra": "mean: 45.808866421057964 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 252.11827363606744,
            "unit": "iter/sec",
            "range": "stddev: 0.00003081679425846564",
            "extra": "mean: 3.966392382344722 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.755360885025045,
            "unit": "iter/sec",
            "range": "stddev: 0.00190456385820529",
            "extra": "mean: 33.607389399981 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.94658933250289,
            "unit": "iter/sec",
            "range": "stddev: 0.0007791461113964511",
            "extra": "mean: 29.45804040002713 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.31148255825015,
            "unit": "iter/sec",
            "range": "stddev: 0.013015722633531521",
            "extra": "mean: 10.491915277771955 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3375.618905225489,
            "unit": "iter/sec",
            "range": "stddev: 0.000008980479557747058",
            "extra": "mean: 296.2419716431825 usec\nrounds: 2257"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.834648294038857,
            "unit": "iter/sec",
            "range": "stddev: 0.0006374889446055735",
            "extra": "mean: 35.92644640004892 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.504551599670883,
            "unit": "iter/sec",
            "range": "stddev: 0.0006870065426912961",
            "extra": "mean: 35.082116500002975 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d937a9622570893248af74609700f77df5bdd148",
          "message": "feat(interpret): propagate const values through `squeeze` (#84)\n\n* feat(interpret): propagate const values through `squeeze`\n\n`prop_squeeze` did not forward const values through `state_consts`,\nbreaking `slice → squeeze → lt → select_n → gather` const chains.\nThis caused MSS1 to detect 6570 nnz instead of the true 378.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: update PLAN.md after scatter-add task\n\nRemove completed scatter-add item, update CUTEst tables\nto reflect resolved problems (GENROSE, COATING, ERRINROS,\nLUKSAN17LS, LUKSAN21LS, MSS1), and note OET2/4/6/7\nas fixture mismatches rather than asdex issues.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-05T22:44:46+01:00",
          "tree_id": "6d4419a94376bb9a34251a6463b32362aaebce6a",
          "url": "https://github.com/adrhill/asdex/commit/d937a9622570893248af74609700f77df5bdd148"
        },
        "date": 1772747119994,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 819.0296784888466,
            "unit": "iter/sec",
            "range": "stddev: 0.0032410423271831902",
            "extra": "mean: 1.2209569766080435 msec\nrounds: 171"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3372.80348597664,
            "unit": "iter/sec",
            "range": "stddev: 0.000008296594441426233",
            "extra": "mean: 296.4892571292029 usec\nrounds: 2104"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 70.16470856610646,
            "unit": "iter/sec",
            "range": "stddev: 0.0005474629587024829",
            "extra": "mean: 14.25217920000108 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 124.96879529178929,
            "unit": "iter/sec",
            "range": "stddev: 0.0002838835116327101",
            "extra": "mean: 8.001997600001687 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.514920907672174,
            "unit": "iter/sec",
            "range": "stddev: 0.00988760227045866",
            "extra": "mean: 44.41499057894716 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.29400643501842,
            "unit": "iter/sec",
            "range": "stddev: 0.00017809349453325171",
            "extra": "mean: 4.027483443349698 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.32096257591592,
            "unit": "iter/sec",
            "range": "stddev: 0.000625302536404291",
            "extra": "mean: 28.31179919999869 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.82287342708464,
            "unit": "iter/sec",
            "range": "stddev: 0.00074661976267406",
            "extra": "mean: 29.565790799998126 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 99.25981056896671,
            "unit": "iter/sec",
            "range": "stddev: 0.011858440203246579",
            "extra": "mean: 10.074570909091046 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3393.567483014415,
            "unit": "iter/sec",
            "range": "stddev: 0.000009190914456765638",
            "extra": "mean: 294.67514791004743 usec\nrounds: 2177"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.220489370038,
            "unit": "iter/sec",
            "range": "stddev: 0.0005978692752609818",
            "extra": "mean: 35.43524660000088 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.55818450937576,
            "unit": "iter/sec",
            "range": "stddev: 0.0003114483594169873",
            "extra": "mean: 35.016231499999456 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f79fa5428a75a39ef639e245396ed9be69756411",
          "message": "feat(interpret): per-element branch selection in `select_n` for const predicates (#85)\n\n* feat(interpret): per-element branch selection in `select_n` for const predicates\n\nWhen the `which` predicate is a known constant, pick index sets\nfrom exactly the selected branch per element instead of unioning\nall branches. Fixes `jnp.diag` precision (was `tile(eye(n), (n,1))`,\nnow exact diagonal).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* claude: update PLAN.md after `select_n` and scatter improvements\n\nRemove completed items (diag, scatter pattern 4).\nFix section counts (Hessians 22→21, eq 13→16, ineq 24→25).\nAdd ELATTAR to ineq small-extras table.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-06T01:57:47+01:00",
          "tree_id": "f5323d25cd7e75d2956e3e65985b02be7de464ac",
          "url": "https://github.com/adrhill/asdex/commit/f79fa5428a75a39ef639e245396ed9be69756411"
        },
        "date": 1772758703620,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 827.1981720153335,
            "unit": "iter/sec",
            "range": "stddev: 0.003104112520924849",
            "extra": "mean: 1.208900156976487 msec\nrounds: 172"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3392.6319322307663,
            "unit": "iter/sec",
            "range": "stddev: 0.000009830869226468109",
            "extra": "mean: 294.75640740740994 usec\nrounds: 2187"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 67.91506048113774,
            "unit": "iter/sec",
            "range": "stddev: 0.0005819219533532165",
            "extra": "mean: 14.724274600001763 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 125.82372071574137,
            "unit": "iter/sec",
            "range": "stddev: 0.0003726930396117812",
            "extra": "mean: 7.947626999992963 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.637196109060454,
            "unit": "iter/sec",
            "range": "stddev: 0.016466939080579217",
            "extra": "mean: 48.4561950526295 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.8597716588334,
            "unit": "iter/sec",
            "range": "stddev: 0.00003766778140222149",
            "extra": "mean: 4.018327242423573 msec\nrounds: 198"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.51565299014559,
            "unit": "iter/sec",
            "range": "stddev: 0.00048563720145213695",
            "extra": "mean: 28.97236219999968 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 32.16665766643269,
            "unit": "iter/sec",
            "range": "stddev: 0.0007007858957335537",
            "extra": "mean: 31.088091600003054 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 99.7166648837775,
            "unit": "iter/sec",
            "range": "stddev: 0.011264738127948846",
            "extra": "mean: 10.028414018514631 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3371.9254904093154,
            "unit": "iter/sec",
            "range": "stddev: 0.000011340857368877923",
            "extra": "mean: 296.56645819852054 usec\nrounds: 2165"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.829125823481174,
            "unit": "iter/sec",
            "range": "stddev: 0.0006500866919832484",
            "extra": "mean: 37.27292519999992 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 27.463308457922167,
            "unit": "iter/sec",
            "range": "stddev: 0.0004450748946934233",
            "extra": "mean: 36.41221892592247 msec\nrounds: 27"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "96b3a072e67b4991bc8cd670901eccfef6876187",
          "message": "claude: reformulate CUTEst section in `PLAN.md` into actionable tasks\n\nReplace 250-line diagnostic report with two `## Later` tasks\n(gather/scatter composition, `reduce_sum` Hessian precision)\nand one `## Ideas` item (algebraic cancellation).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-06T02:04:49+01:00",
          "tree_id": "50f35d851acc0ec0c3eadb7ac7a8f2ab737511f2",
          "url": "https://github.com/adrhill/asdex/commit/96b3a072e67b4991bc8cd670901eccfef6876187"
        },
        "date": 1772759134815,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 793.3822886162069,
            "unit": "iter/sec",
            "range": "stddev: 0.004172389735919537",
            "extra": "mean: 1.2604264228587325 msec\nrounds: 175"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3583.2731375817525,
            "unit": "iter/sec",
            "range": "stddev: 0.000006584911361894186",
            "extra": "mean: 279.0744555618418 usec\nrounds: 1879"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 73.07832185216456,
            "unit": "iter/sec",
            "range": "stddev: 0.0003965863767932273",
            "extra": "mean: 13.683948600009899 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 133.4494770821604,
            "unit": "iter/sec",
            "range": "stddev: 0.0002691507058378936",
            "extra": "mean: 7.49347260000377 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.826526703738658,
            "unit": "iter/sec",
            "range": "stddev: 0.01958048730972921",
            "extra": "mean: 48.01568760001089 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 272.258773356198,
            "unit": "iter/sec",
            "range": "stddev: 0.000031649266203518946",
            "extra": "mean: 3.6729762191784108 msec\nrounds: 219"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 23.437320301675246,
            "unit": "iter/sec",
            "range": "stddev: 0.030390019499359842",
            "extra": "mean: 42.666993799991815 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 36.35382981889251,
            "unit": "iter/sec",
            "range": "stddev: 0.0003950766632030504",
            "extra": "mean: 27.507418200002576 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 97.02465620783676,
            "unit": "iter/sec",
            "range": "stddev: 0.012980100882699595",
            "extra": "mean: 10.30665852459088 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3461.4090284304057,
            "unit": "iter/sec",
            "range": "stddev: 0.00003274930751116714",
            "extra": "mean: 288.8996913645468 usec\nrounds: 2038"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.12735822498233,
            "unit": "iter/sec",
            "range": "stddev: 0.0007122621359041662",
            "extra": "mean: 33.192422400009036 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 31.521537656419405,
            "unit": "iter/sec",
            "range": "stddev: 0.0005925826138263449",
            "extra": "mean: 31.72434069999592 msec\nrounds: 30"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "768a0b13883fcf32902e3ebfe1e3b755f37bf7ba",
          "message": "feat(interpret): per-timestep forward simulation in `scan` handler (#87)\n\n* feat(interpret): per-timestep forward simulation in `scan` handler\n\nReplace the 3-stage approach (merge xs → fixed-point → tile ys)\nwith forward simulation: one `prop_jaxpr` call per timestep,\nthreading carry deps forward and collecting per-timestep ys.\n\nProduces exact per-timestep sparsity patterns instead of\noverapproximate all-to-all dependencies (e.g. cumulative sum\nyields lower-triangular instead of dense).\n\nRemoves 9 `@pytest.mark.fallback` markers and `TODO(scan)` comments.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* claude: remove completed `scan` task from `PLAN.md`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): address review feedback on `scan` handler\n\nUse `atom_shape` helper instead of raw `getattr(…, \"shape\", ())`.\nMove pattern explanations from docstrings to inline comments\non expected matrices.\nDocument inline matrix comment style in `tests/CLAUDE.md`\nand add a task in `PLAN.md` to apply it across all test files.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): move `fixed_point_loop` into `_while.py`\n\nOnly used by the while handler, so inline it as `_fixed_point_loop`\nrather than keeping it in the shared `_commons.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* claude: add transfer matrix design notes and cross-task insights to `PLAN.md`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): satisfy `ty` type checker for `list[int | slice]` in gather/scatter\n\n`[0] * n` is inferred as `list[int]` by `ty`, which rejects\nassignment to `list[int | slice]`. Using a list comprehension\nlets `ty` infer the wider type from the annotation.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-06T17:42:44+01:00",
          "tree_id": "b99e9d3a246f519492bd96bb4d7a6abf9a0bebba",
          "url": "https://github.com/adrhill/asdex/commit/768a0b13883fcf32902e3ebfe1e3b755f37bf7ba"
        },
        "date": 1772815397837,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 715.0155089098738,
            "unit": "iter/sec",
            "range": "stddev: 0.0036665043367069927",
            "extra": "mean: 1.3985710624999153 msec\nrounds: 144"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3397.6512797386663,
            "unit": "iter/sec",
            "range": "stddev: 0.000007882447354981062",
            "extra": "mean: 294.3209640033794 usec\nrounds: 1139"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 67.81424647114865,
            "unit": "iter/sec",
            "range": "stddev: 0.0003320302360454696",
            "extra": "mean: 14.746164000000306 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 115.9020887897807,
            "unit": "iter/sec",
            "range": "stddev: 0.000293924833873424",
            "extra": "mean: 8.627972199998624 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.07534254924843,
            "unit": "iter/sec",
            "range": "stddev: 0.012182930216457286",
            "extra": "mean: 45.299410315789 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 251.41777516690647,
            "unit": "iter/sec",
            "range": "stddev: 0.00007613842621186065",
            "extra": "mean: 3.977443517412955 msec\nrounds: 201"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 32.83421345005038,
            "unit": "iter/sec",
            "range": "stddev: 0.0007907011506887326",
            "extra": "mean: 30.456036399996833 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 32.4357170309374,
            "unit": "iter/sec",
            "range": "stddev: 0.0007942772022579432",
            "extra": "mean: 30.83021099999712 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 90.93400666794481,
            "unit": "iter/sec",
            "range": "stddev: 0.014866590342788496",
            "extra": "mean: 10.996986019230475 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3356.0540253392664,
            "unit": "iter/sec",
            "range": "stddev: 0.000008560552084323042",
            "extra": "mean: 297.9689815627772 usec\nrounds: 2278"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.4101904755231,
            "unit": "iter/sec",
            "range": "stddev: 0.0005499693488272188",
            "extra": "mean: 37.86417219999976 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 26.490142173470062,
            "unit": "iter/sec",
            "range": "stddev: 0.0004859145717695317",
            "extra": "mean: 37.74989176922962 msec\nrounds: 26"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "767c2d729de4e03c15b9ab6d6710c12294db1260",
          "message": "asdex `v0.1.3`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-06T17:45:21+01:00",
          "tree_id": "aeec708cd3a575a31f71f43231f7611269003b1f",
          "url": "https://github.com/adrhill/asdex/commit/767c2d729de4e03c15b9ab6d6710c12294db1260"
        },
        "date": 1772815569350,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 850.0503303073465,
            "unit": "iter/sec",
            "range": "stddev: 0.0028194277870314797",
            "extra": "mean: 1.176400931034798 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3398.501670741385,
            "unit": "iter/sec",
            "range": "stddev: 0.000006954897007805857",
            "extra": "mean: 294.24731746029994 usec\nrounds: 2142"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.9369570086016,
            "unit": "iter/sec",
            "range": "stddev: 0.0005023161097908886",
            "extra": "mean: 14.29859179999795 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 126.0339732641311,
            "unit": "iter/sec",
            "range": "stddev: 0.00030786814116927987",
            "extra": "mean: 7.93436859999872 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.755447457356976,
            "unit": "iter/sec",
            "range": "stddev: 0.01390230357612161",
            "extra": "mean: 48.180122449999985 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.75339633488667,
            "unit": "iter/sec",
            "range": "stddev: 0.00048023005743998544",
            "extra": "mean: 4.136446540816159 msec\nrounds: 196"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 26.27925743311914,
            "unit": "iter/sec",
            "range": "stddev: 0.022397880982963178",
            "extra": "mean: 38.05282560000052 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 33.41463916994617,
            "unit": "iter/sec",
            "range": "stddev: 0.0007276888214263645",
            "extra": "mean: 29.927002800000935 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 104.32824427235555,
            "unit": "iter/sec",
            "range": "stddev: 0.009496661569856764",
            "extra": "mean: 9.585132070174938 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3390.679022749627,
            "unit": "iter/sec",
            "range": "stddev: 0.000008651910372833644",
            "extra": "mean: 294.9261765240943 usec\nrounds: 2198"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.73892909936326,
            "unit": "iter/sec",
            "range": "stddev: 0.0006504616105209566",
            "extra": "mean: 36.050418400000694 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.50192325786732,
            "unit": "iter/sec",
            "range": "stddev: 0.00013665231426779388",
            "extra": "mean: 35.08535164285703 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "4dfb4562602e51fb02e76952eb1b69f4809aa77a",
          "message": "claude: minor fixes to release skill",
          "timestamp": "2026-03-06T17:50:54+01:00",
          "tree_id": "ba6512e116e072ea546bf4d52833970c9b929fef",
          "url": "https://github.com/adrhill/asdex/commit/4dfb4562602e51fb02e76952eb1b69f4809aa77a"
        },
        "date": 1772815895539,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 801.4321021759661,
            "unit": "iter/sec",
            "range": "stddev: 0.0041582853770161295",
            "extra": "mean: 1.2477663388887252 msec\nrounds: 180"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3553.127625375977,
            "unit": "iter/sec",
            "range": "stddev: 0.000009795418554103366",
            "extra": "mean: 281.44218430492884 usec\nrounds: 2230"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 71.98660335070375,
            "unit": "iter/sec",
            "range": "stddev: 0.0007567616932835929",
            "extra": "mean: 13.891473600000381 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 137.8817698018849,
            "unit": "iter/sec",
            "range": "stddev: 0.0002049740827599812",
            "extra": "mean: 7.252590399998837 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.626787395858035,
            "unit": "iter/sec",
            "range": "stddev: 0.013896391872555378",
            "extra": "mean: 44.19540355000002 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 268.02329576294215,
            "unit": "iter/sec",
            "range": "stddev: 0.00004081262270259998",
            "extra": "mean: 3.731018966666492 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.13426216392763,
            "unit": "iter/sec",
            "range": "stddev: 0.0008181328371368481",
            "extra": "mean: 29.29607779999941 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 35.10456505521394,
            "unit": "iter/sec",
            "range": "stddev: 0.00019925788529324482",
            "extra": "mean: 28.48632359999783 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.8644588898874,
            "unit": "iter/sec",
            "range": "stddev: 0.0128874425696388",
            "extra": "mean: 10.323703982456248 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3514.1873476894857,
            "unit": "iter/sec",
            "range": "stddev: 0.000006497222568153202",
            "extra": "mean: 284.5608105263602 usec\nrounds: 1900"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.21315787738461,
            "unit": "iter/sec",
            "range": "stddev: 0.0006936745794603662",
            "extra": "mean: 33.098162200003856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 31.15912001227509,
            "unit": "iter/sec",
            "range": "stddev: 0.00057220553722714",
            "extra": "mean: 32.093332533333786 msec\nrounds: 30"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "141c3197fff2cff4f55f9e364cd0a2419288e1c4",
          "message": "fix(interpret): handle zero-sized arrays (#88)\n\n* test(interpret): add size-0 dimension tests across all handlers\n\nDocument zero-sized array behavior for 16 handler test files.\nThree handlers crash on zero-sized inputs (`broadcast_in_dim`,\n`sort`, `gather`) — these are marked `@pytest.mark.bug`.\nThe remaining 13 handlers already handle them correctly.\n\nCovers issue #86.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): handle zero-sized arrays in `broadcast_in_dim`, `sort`, and `gather` (#86)\n\nAdd early returns when the output has zero elements,\navoiding `np.ravel_multi_index` and `np.reshape` crashes\non zero-sized shapes.\nUpdate the corresponding tests from `@pytest.mark.bug`\nwith `pytest.raises` to `@pytest.mark.array_ops`\nasserting correct behavior.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* claude: document zero-sized array handling in `CLAUDE.md` files and `add-handler` skill\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-09T12:01:42+01:00",
          "tree_id": "4fc5148634ecb5dbe47b155cf0868144e7a2c9d7",
          "url": "https://github.com/adrhill/asdex/commit/141c3197fff2cff4f55f9e364cd0a2419288e1c4"
        },
        "date": 1773054137942,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 728.9306331409772,
            "unit": "iter/sec",
            "range": "stddev: 0.0050770624839236965",
            "extra": "mean: 1.3718726508872037 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3332.3513708310184,
            "unit": "iter/sec",
            "range": "stddev: 0.000011254150979272422",
            "extra": "mean: 300.08840266763974 usec\nrounds: 2024"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 62.596373376453116,
            "unit": "iter/sec",
            "range": "stddev: 0.00041681992317460384",
            "extra": "mean: 15.975366399999302 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 115.32695988885986,
            "unit": "iter/sec",
            "range": "stddev: 0.0002819503038887047",
            "extra": "mean: 8.670999399998891 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.889705654717226,
            "unit": "iter/sec",
            "range": "stddev: 0.017559439281053288",
            "extra": "mean: 47.87046866666521 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.18423649807866,
            "unit": "iter/sec",
            "range": "stddev: 0.000037619985067428445",
            "extra": "mean: 4.013094945545284 msec\nrounds: 202"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.673864203898614,
            "unit": "iter/sec",
            "range": "stddev: 0.0005541460009474547",
            "extra": "mean: 31.571771399995896 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 30.798537710068917,
            "unit": "iter/sec",
            "range": "stddev: 0.00014841332418883454",
            "extra": "mean: 32.46907399999941 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 84.9212614309976,
            "unit": "iter/sec",
            "range": "stddev: 0.0164122404387954",
            "extra": "mean: 11.775614058824898 msec\nrounds: 51"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3323.9181838132763,
            "unit": "iter/sec",
            "range": "stddev: 0.000009859822209120436",
            "extra": "mean: 300.8497636523582 usec\nrounds: 2234"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.51040711629109,
            "unit": "iter/sec",
            "range": "stddev: 0.0005322470268303999",
            "extra": "mean: 37.72103520000201 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 26.532121661912132,
            "unit": "iter/sec",
            "range": "stddev: 0.0005651366917499184",
            "extra": "mean: 37.69016337036996 msec\nrounds: 27"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "c24c43de2f5bcc05da2aa798c82805d110e65bf8",
          "message": "asdex `v0.1.4`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-09T12:03:05+01:00",
          "tree_id": "d8980bade599de919ecf290cfb93d1425b35d9e2",
          "url": "https://github.com/adrhill/asdex/commit/c24c43de2f5bcc05da2aa798c82805d110e65bf8"
        },
        "date": 1773054227517,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 756.6701869620239,
            "unit": "iter/sec",
            "range": "stddev: 0.004342902241855913",
            "extra": "mean: 1.3215797546021046 msec\nrounds: 163"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3190.6164501186713,
            "unit": "iter/sec",
            "range": "stddev: 0.00005171235245682014",
            "extra": "mean: 313.41905729935235 usec\nrounds: 2007"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 66.92243260420362,
            "unit": "iter/sec",
            "range": "stddev: 0.0006303365907692655",
            "extra": "mean: 14.942672599997309 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 117.78889605494062,
            "unit": "iter/sec",
            "range": "stddev: 0.00045206871796518846",
            "extra": "mean: 8.48976459999733 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.291497572456922,
            "unit": "iter/sec",
            "range": "stddev: 0.018808561859474222",
            "extra": "mean: 49.28172484210186 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.17799911929941,
            "unit": "iter/sec",
            "range": "stddev: 0.000037613935132305765",
            "extra": "mean: 3.997154040404415 msec\nrounds: 198"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 33.20837663235226,
            "unit": "iter/sec",
            "range": "stddev: 0.0005431540402123631",
            "extra": "mean: 30.112884199999712 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 31.792032293584636,
            "unit": "iter/sec",
            "range": "stddev: 0.0010238647971543547",
            "extra": "mean: 31.454421999998775 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 106.59224073411106,
            "unit": "iter/sec",
            "range": "stddev: 0.009472656952748695",
            "extra": "mean: 9.381545909091537 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3303.99975305738,
            "unit": "iter/sec",
            "range": "stddev: 0.000007464080319211435",
            "extra": "mean: 302.66346087787775 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.997153041372314,
            "unit": "iter/sec",
            "range": "stddev: 0.0010245722150300362",
            "extra": "mean: 38.46575039999891 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 26.821140204013236,
            "unit": "iter/sec",
            "range": "stddev: 0.0004909305537812776",
            "extra": "mean: 37.28402269230786 msec\nrounds: 26"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ee114cce53a30051f750eec3ae362167821c72d9",
          "message": "feat: add `value_and_jacobian` / `value_and_hessian` API (#89)\n\n* feat: add `value_and_jacobian` / `value_and_hessian` API\n\nReturn `f(x)` alongside the sparse derivative without an extra\nforward pass. The primal is extracted for free from VJP, JVP,\nand HVP computations already performed internally.\n\nNew public functions:\n- `value_and_jacobian`, `value_and_jacobian_from_coloring`\n- `value_and_hessian`, `value_and_hessian_from_coloring`\n\nRefactored `_jacobian_rows`, `_jacobian_cols`, and `_compute_hvps`\nto return `(value, result)` tuples.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add `value_and_jacobian` / `value_and_hessian` to docs\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf: split inner-loop helpers into lean and `value_and_` variants\n\nThe `value_and_jacobian` / `value_and_hessian` PR modified\n`_jacobian_rows`, `_jacobian_cols`, and `_compute_hvps` to always\nextract the primal, penalizing the non-`value_and_` paths:\n\n- `_jacobian_cols`: primals materialized num_colors times under vmap\n- `_compute_hvps`: `value_and_grad` adds tuple cotangents and extra\n  residuals; `rev_over_fwd` gained a gratuitous `f(x)` call\n\nRestore the original lean implementations for the Jacobian-only and\nHessian-only paths, and add `_value_and_` variants that pay for\nprimal extraction only when the caller needs it.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf: avoid redundant primal computation in fwd-mode Jacobian and `rev_over_rev` value path\n\nUse `jax.linearize` in `_jacobian_cols` and `_value_and_jacobian_cols`\nso the nonlinear forward pass runs once instead of once per color.\n\nIn `_value_and_compute_hvps` `rev_over_rev`, replace\n`jax.vjp(jax.value_and_grad(f), x)` with a separate `f(x)` call and\n`jax.vjp(jax.grad(f), x)`, eliminating the zero-cotangent overhead.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: add `value_and_*` materialization benchmarks\n\nAdd `value_and_jacobian_from_coloring` and `value_and_hessian_from_coloring`\nbenchmarks for heat equation, convnet, and rosenbrock groups.\n\nAlso document the redundant `f(x)` in the `rev_over_rev` value path\nwith a TODO explaining the tradeoff.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix: rename `vjp_fn` to `hvp_fn` in `rev_over_rev` and fix Hessian docstrings\n\nThe VJP of `grad(f)` is semantically an HVP — rename accordingly\nand use tuple unpacking for clarity.\n\nAlso fix `value_and_hessian` docstrings: \"but also returns\" →\n\"but can also return\", since `rev_over_fwd` needs an extra `f(x)`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: cover shape mismatch guards for `value_and_*` variants\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-10T14:43:05+01:00",
          "tree_id": "c3607ce3e6014db072a9407265b5bd2104471fb4",
          "url": "https://github.com/adrhill/asdex/commit/ee114cce53a30051f750eec3ae362167821c72d9"
        },
        "date": 1773150221136,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 811.0052367212508,
            "unit": "iter/sec",
            "range": "stddev: 0.003384858726693907",
            "extra": "mean: 1.2330376608205653 msec\nrounds: 171"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3401.8616036488083,
            "unit": "iter/sec",
            "range": "stddev: 0.000011370102946859441",
            "extra": "mean: 293.95669680606886 usec\nrounds: 2223"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.61478990163215,
            "unit": "iter/sec",
            "range": "stddev: 0.0003737617794659263",
            "extra": "mean: 14.364763599991193 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 70.95491937920117,
            "unit": "iter/sec",
            "range": "stddev: 0.00012537219251824977",
            "extra": "mean: 14.093455517238278 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 85.37917051848515,
            "unit": "iter/sec",
            "range": "stddev: 0.0002966741754448619",
            "extra": "mean: 11.712458599998854 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.097165162945434,
            "unit": "iter/sec",
            "range": "stddev: 0.01603751233119335",
            "extra": "mean: 47.39973320000246 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 252.1677697659222,
            "unit": "iter/sec",
            "range": "stddev: 0.000031150310848366224",
            "extra": "mean: 3.965613848781159 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 25.10342094810654,
            "unit": "iter/sec",
            "range": "stddev: 0.025494969338535366",
            "extra": "mean: 39.835208199997396 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 35.01695096185056,
            "unit": "iter/sec",
            "range": "stddev: 0.000302762479476824",
            "extra": "mean: 28.557597749999886 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 39.93096224130816,
            "unit": "iter/sec",
            "range": "stddev: 0.0008338495185974314",
            "extra": "mean: 25.043223200003695 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 101.0586747197221,
            "unit": "iter/sec",
            "range": "stddev: 0.010884020859773018",
            "extra": "mean: 9.895241578948244 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3274.609184417725,
            "unit": "iter/sec",
            "range": "stddev: 0.000010599783458140878",
            "extra": "mean: 305.37995335703397 usec\nrounds: 2294"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.5623914696367,
            "unit": "iter/sec",
            "range": "stddev: 0.0010943633884018928",
            "extra": "mean: 36.28132200000209 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 28.48126191433271,
            "unit": "iter/sec",
            "range": "stddev: 0.00033885528748281874",
            "extra": "mean: 35.110803833336014 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.478253074308448,
            "unit": "iter/sec",
            "range": "stddev: 0.00032141788620501844",
            "extra": "mean: 35.11451342856934 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "fe26c83535d426fe4e16d966c9342fba0471d583",
          "message": "asdex `v0.1.5`",
          "timestamp": "2026-03-10T14:45:11+01:00",
          "tree_id": "c3e8f65fe60ba403c09a006c644eec0a5d9034d5",
          "url": "https://github.com/adrhill/asdex/commit/fe26c83535d426fe4e16d966c9342fba0471d583"
        },
        "date": 1773150353332,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 855.4986087430958,
            "unit": "iter/sec",
            "range": "stddev: 0.0026960035094264006",
            "extra": "mean: 1.168908972826042 msec\nrounds: 184"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3360.7151680655493,
            "unit": "iter/sec",
            "range": "stddev: 0.000013665846115067877",
            "extra": "mean: 297.55571358807146 usec\nrounds: 2252"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.47299977715312,
            "unit": "iter/sec",
            "range": "stddev: 0.0005040272501236094",
            "extra": "mean: 14.394081200001096 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 71.90547620838568,
            "unit": "iter/sec",
            "range": "stddev: 0.00011296465624492518",
            "extra": "mean: 13.90714661428498 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 86.51664703142616,
            "unit": "iter/sec",
            "range": "stddev: 0.00020654990502820574",
            "extra": "mean: 11.558469199999877 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.70816740995485,
            "unit": "iter/sec",
            "range": "stddev: 0.01291637034167242",
            "extra": "mean: 46.0656112105264 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 253.1086451565849,
            "unit": "iter/sec",
            "range": "stddev: 0.00004517954132558116",
            "extra": "mean: 3.9508725566499434 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.73846738241027,
            "unit": "iter/sec",
            "range": "stddev: 0.0006283953512385024",
            "extra": "mean: 27.98105440000427 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 36.075078193387974,
            "unit": "iter/sec",
            "range": "stddev: 0.0001331999043792583",
            "extra": "mean: 27.719967636363577 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.746362199416836,
            "unit": "iter/sec",
            "range": "stddev: 0.001084412889732052",
            "extra": "mean: 24.542068200000244 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 106.64325788110206,
            "unit": "iter/sec",
            "range": "stddev: 0.008556887954960254",
            "extra": "mean: 9.377057864407265 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3232.6679626652576,
            "unit": "iter/sec",
            "range": "stddev: 0.000040385177609696156",
            "extra": "mean: 309.34200838106614 usec\nrounds: 2267"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.95549270862571,
            "unit": "iter/sec",
            "range": "stddev: 0.0004390144025827047",
            "extra": "mean: 35.77114560000041 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 28.41177353860237,
            "unit": "iter/sec",
            "range": "stddev: 0.0005282123820372839",
            "extra": "mean: 35.19667642857018 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.465570343691503,
            "unit": "iter/sec",
            "range": "stddev: 0.00029429954387639497",
            "extra": "mean: 35.130158571427266 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "8edd6f0182233af3a715fbfee81cab8185fd3029",
          "message": "docs: add logo to docs and README",
          "timestamp": "2026-03-13T14:56:08+01:00",
          "tree_id": "87e1b9364b3e7e99832c3a746fdc11611512d636",
          "url": "https://github.com/adrhill/asdex/commit/8edd6f0182233af3a715fbfee81cab8185fd3029"
        },
        "date": 1773410214797,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 848.0099094370295,
            "unit": "iter/sec",
            "range": "stddev: 0.0027627296847243603",
            "extra": "mean: 1.1792315029241494 msec\nrounds: 171"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3402.178656020285,
            "unit": "iter/sec",
            "range": "stddev: 0.000010616495738049108",
            "extra": "mean: 293.9293026926913 usec\nrounds: 2154"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.86199723837683,
            "unit": "iter/sec",
            "range": "stddev: 0.00047720548653329675",
            "extra": "mean: 14.521797800001934 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 71.00139235994067,
            "unit": "iter/sec",
            "range": "stddev: 0.00011149648972305413",
            "extra": "mean: 14.084230840580034 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.97890594110227,
            "unit": "iter/sec",
            "range": "stddev: 0.0003961313299856789",
            "extra": "mean: 11.767626200001757 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.736478087054415,
            "unit": "iter/sec",
            "range": "stddev: 0.01394035558107021",
            "extra": "mean: 46.00561305263016 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 251.1073478816401,
            "unit": "iter/sec",
            "range": "stddev: 0.000033669944965555206",
            "extra": "mean: 3.9823605658538983 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.03988211350399,
            "unit": "iter/sec",
            "range": "stddev: 0.000990263091574916",
            "extra": "mean: 28.538908799998808 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 35.71165940000932,
            "unit": "iter/sec",
            "range": "stddev: 0.0001366077769008705",
            "extra": "mean: 28.00205918181834 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.003499506138695,
            "unit": "iter/sec",
            "range": "stddev: 0.0007119188419195763",
            "extra": "mean: 24.997812999998814 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 98.5348362523466,
            "unit": "iter/sec",
            "range": "stddev: 0.011749923797588427",
            "extra": "mean: 10.148694999999913 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3316.2166846925743,
            "unit": "iter/sec",
            "range": "stddev: 0.000009783146350520176",
            "extra": "mean: 301.54844965829 usec\nrounds: 2195"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.453330504909527,
            "unit": "iter/sec",
            "range": "stddev: 0.0007141956100633997",
            "extra": "mean: 36.42545299999824 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 27.798149009784566,
            "unit": "iter/sec",
            "range": "stddev: 0.00031214320146256166",
            "extra": "mean: 35.97361823076831 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.074597518409373,
            "unit": "iter/sec",
            "range": "stddev: 0.0001816667487554277",
            "extra": "mean: 35.61938864285657 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "db66d79d70cfc712da068df65af1c77d91161fc1",
          "message": "docs: move all files into `docs/`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-13T23:34:38+01:00",
          "tree_id": "53106848343129e67b0e3a68312a225075201c83",
          "url": "https://github.com/adrhill/asdex/commit/db66d79d70cfc712da068df65af1c77d91161fc1"
        },
        "date": 1773441326670,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 817.0243217871184,
            "unit": "iter/sec",
            "range": "stddev: 0.003378235202558012",
            "extra": "mean: 1.223953771428309 msec\nrounds: 175"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3278.276922511502,
            "unit": "iter/sec",
            "range": "stddev: 0.00004516795304822662",
            "extra": "mean: 305.0382940907553 usec\nrounds: 2183"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.9594421523061,
            "unit": "iter/sec",
            "range": "stddev: 0.0005891917500642829",
            "extra": "mean: 14.501277400002266 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 70.83513741573977,
            "unit": "iter/sec",
            "range": "stddev: 0.0004289696400828672",
            "extra": "mean: 14.117287499999927 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 72.88350952873887,
            "unit": "iter/sec",
            "range": "stddev: 0.0013623104709863877",
            "extra": "mean: 13.720524799998657 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.071806064947214,
            "unit": "iter/sec",
            "range": "stddev: 0.014298120625572156",
            "extra": "mean: 45.306668473683494 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 251.82283968449224,
            "unit": "iter/sec",
            "range": "stddev: 0.00016740199404368078",
            "extra": "mean: 3.971045681372252 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.624385490735335,
            "unit": "iter/sec",
            "range": "stddev: 0.0006607148832215009",
            "extra": "mean: 28.881378999999185 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 35.09489307572749,
            "unit": "iter/sec",
            "range": "stddev: 0.0002797919820301792",
            "extra": "mean: 28.494174290322174 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.25643899124943,
            "unit": "iter/sec",
            "range": "stddev: 0.0003809112172814182",
            "extra": "mean: 24.84074660000033 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 100.55484310097658,
            "unit": "iter/sec",
            "range": "stddev: 0.01126683729929827",
            "extra": "mean: 9.94482184210467 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3316.9536972632204,
            "unit": "iter/sec",
            "range": "stddev: 0.000011119618181359465",
            "extra": "mean: 301.48144691470617 usec\nrounds: 2204"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 27.034977794792635,
            "unit": "iter/sec",
            "range": "stddev: 0.0005647708051393621",
            "extra": "mean: 36.9891186000018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 27.66133943810462,
            "unit": "iter/sec",
            "range": "stddev: 0.00046952539642932387",
            "extra": "mean: 36.151539307690186 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 27.911932172368555,
            "unit": "iter/sec",
            "range": "stddev: 0.0004428979656448349",
            "extra": "mean: 35.82697155555397 msec\nrounds: 27"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "960bbc4f46ff2b7336f76869494d9ce92c02e496",
          "message": "docs: add color logo",
          "timestamp": "2026-03-23T16:22:20+01:00",
          "tree_id": "fc0b50c90dd2a3b403c0de9ab94fce776ff7f772",
          "url": "https://github.com/adrhill/asdex/commit/960bbc4f46ff2b7336f76869494d9ce92c02e496"
        },
        "date": 1774279380191,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 832.9828063493711,
            "unit": "iter/sec",
            "range": "stddev: 0.0032203572102258698",
            "extra": "mean: 1.2005049712641707 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3327.042966336089,
            "unit": "iter/sec",
            "range": "stddev: 0.00002322854801687067",
            "extra": "mean: 300.5672034050259 usec\nrounds: 2232"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.81318703244281,
            "unit": "iter/sec",
            "range": "stddev: 0.00043833733257236067",
            "extra": "mean: 14.323941400000706 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 70.86726354563605,
            "unit": "iter/sec",
            "range": "stddev: 0.0004223894583970155",
            "extra": "mean: 14.110887735294519 msec\nrounds: 68"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.29733517915882,
            "unit": "iter/sec",
            "range": "stddev: 0.00029092833149584285",
            "extra": "mean: 11.862771199999145 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.758873340031506,
            "unit": "iter/sec",
            "range": "stddev: 0.01947400514174828",
            "extra": "mean: 50.61017310000153 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 254.30335524193023,
            "unit": "iter/sec",
            "range": "stddev: 0.000035737332772118035",
            "extra": "mean: 3.932311467336539 msec\nrounds: 199"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 33.86718080117975,
            "unit": "iter/sec",
            "range": "stddev: 0.0015014974991779157",
            "extra": "mean: 29.527110799998013 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 35.01486097446763,
            "unit": "iter/sec",
            "range": "stddev: 0.0009378635935914623",
            "extra": "mean: 28.55930231250059 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 38.63123716583269,
            "unit": "iter/sec",
            "range": "stddev: 0.000933141545991269",
            "extra": "mean: 25.885787600000754 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 106.94573525369104,
            "unit": "iter/sec",
            "range": "stddev: 0.010094376399525073",
            "extra": "mean: 9.350536490565544 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3393.7749455115636,
            "unit": "iter/sec",
            "range": "stddev: 0.00000773832927127347",
            "extra": "mean: 294.65713432841204 usec\nrounds: 2010"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.665216110773507,
            "unit": "iter/sec",
            "range": "stddev: 0.00039635986476251793",
            "extra": "mean: 33.70951339999948 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.437378835673186,
            "unit": "iter/sec",
            "range": "stddev: 0.00036012361774574237",
            "extra": "mean: 32.85434023076853 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 29.387374234816207,
            "unit": "iter/sec",
            "range": "stddev: 0.000479080045901542",
            "extra": "mean: 34.028218785714664 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0da5b8b065de3d0970600f37a5ecbe8850d9b3f9",
          "message": "feat: validate and coerce inputs in `*_coloring_from_sparsity` (#90)\n\n* fix: validate inputs in `*_coloring_from_sparsity` functions\n\n`jacobian_coloring_from_sparsity` and `hessian_coloring_from_sparsity`\nnow raise `TypeError` when passed a non-`SparsityPattern` object,\nwith a message pointing to the correct detection function.\n`hessian_coloring_from_sparsity` also validates that the pattern is\nsquare. Docstrings updated to use explicit cross-references.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat: accept NumPy arrays and JAX BCOO in `*_coloring_from_sparsity`\n\nAdd `_coerce_sparsity()` helper that auto-converts `np.ndarray`\n(via `from_dense`) and `BCOO` (via `from_bcoo`) to `SparsityPattern`,\nraising `TypeError` for unsupported types.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-23T18:27:23+01:00",
          "tree_id": "c1c0dbe9a22e705ee4902b6d9a82f15eb23bb20f",
          "url": "https://github.com/adrhill/asdex/commit/0da5b8b065de3d0970600f37a5ecbe8850d9b3f9"
        },
        "date": 1774286883989,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 795.2526781072443,
            "unit": "iter/sec",
            "range": "stddev: 0.0033551933115579485",
            "extra": "mean: 1.2574619709298789 msec\nrounds: 172"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 2868.5766471118977,
            "unit": "iter/sec",
            "range": "stddev: 0.000008490965830172106",
            "extra": "mean: 348.60494350283676 usec\nrounds: 1947"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 70.36983415223136,
            "unit": "iter/sec",
            "range": "stddev: 0.00038020384702374847",
            "extra": "mean: 14.21063460000056 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 71.00932563140287,
            "unit": "iter/sec",
            "range": "stddev: 0.0005539385406497022",
            "extra": "mean: 14.082657328571559 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 85.20948573124848,
            "unit": "iter/sec",
            "range": "stddev: 0.00036050715981743173",
            "extra": "mean: 11.73578260000312 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.006960825284196,
            "unit": "iter/sec",
            "range": "stddev: 0.016705885203495825",
            "extra": "mean: 52.61230394444441 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 199.20848889524055,
            "unit": "iter/sec",
            "range": "stddev: 0.0005230051701879472",
            "extra": "mean: 5.019866400000045 msec\nrounds: 175"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.29144703946262,
            "unit": "iter/sec",
            "range": "stddev: 0.0010191095422627822",
            "extra": "mean: 28.335477400000286 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 33.92348509211079,
            "unit": "iter/sec",
            "range": "stddev: 0.0005305306653340229",
            "extra": "mean: 29.4781033636358 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 39.106587483548054,
            "unit": "iter/sec",
            "range": "stddev: 0.0007623920009850064",
            "extra": "mean: 25.571139399997378 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.70831733845827,
            "unit": "iter/sec",
            "range": "stddev: 0.011676421327072315",
            "extra": "mean: 10.448412716980995 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 2837.7033886032077,
            "unit": "iter/sec",
            "range": "stddev: 0.00001399138582896322",
            "extra": "mean: 352.39764804743254 usec\nrounds: 2023"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.80518188600407,
            "unit": "iter/sec",
            "range": "stddev: 0.0004653039895717708",
            "extra": "mean: 33.551212799999064 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.881183134710767,
            "unit": "iter/sec",
            "range": "stddev: 0.00036735739151640736",
            "extra": "mean: 32.38217900000048 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 30.74625514482131,
            "unit": "iter/sec",
            "range": "stddev: 0.00019864830607641118",
            "extra": "mean: 32.52428613793096 msec\nrounds: 29"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "669b5c5fe6604a5eda290464694c2c51bc0d779a",
          "message": "feat: export `color_rows`, `color_cols`, and `color_symmetric` in public API\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-23T19:36:16+01:00",
          "tree_id": "9b54e38bf279c216c42cca2298a1a50af359ec02",
          "url": "https://github.com/adrhill/asdex/commit/669b5c5fe6604a5eda290464694c2c51bc0d779a"
        },
        "date": 1774291026225,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 831.5525418575966,
            "unit": "iter/sec",
            "range": "stddev: 0.003229037329270654",
            "extra": "mean: 1.2025698313255233 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3370.0146871506045,
            "unit": "iter/sec",
            "range": "stddev: 0.000007711719211796829",
            "extra": "mean: 296.7346118142631 usec\nrounds: 2133"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 71.64385199245498,
            "unit": "iter/sec",
            "range": "stddev: 0.0005079595246534664",
            "extra": "mean: 13.957931800000267 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 73.30668379482844,
            "unit": "iter/sec",
            "range": "stddev: 0.00023989682896400562",
            "extra": "mean: 13.641320930555404 msec\nrounds: 72"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 86.2130905163379,
            "unit": "iter/sec",
            "range": "stddev: 0.000473003175979965",
            "extra": "mean: 11.599166600001354 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.41166124633724,
            "unit": "iter/sec",
            "range": "stddev: 0.012084818455698884",
            "extra": "mean: 44.6196285500001 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.88325606644068,
            "unit": "iter/sec",
            "range": "stddev: 0.0005161371310175857",
            "extra": "mean: 4.001868775609812 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.78949478117548,
            "unit": "iter/sec",
            "range": "stddev: 0.0006860488602469473",
            "extra": "mean: 27.94116000000031 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 36.75660508112301,
            "unit": "iter/sec",
            "range": "stddev: 0.00021528330266263804",
            "extra": "mean: 27.205994617646756 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.26471732392563,
            "unit": "iter/sec",
            "range": "stddev: 0.0010971055359289552",
            "extra": "mean: 24.83563940000124 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 102.77920943506453,
            "unit": "iter/sec",
            "range": "stddev: 0.010329551884995049",
            "extra": "mean: 9.729594200000106 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3367.828602735556,
            "unit": "iter/sec",
            "range": "stddev: 0.000010064104036918737",
            "extra": "mean: 296.92722461818244 usec\nrounds: 2226"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.923692370103897,
            "unit": "iter/sec",
            "range": "stddev: 0.0005371197130924478",
            "extra": "mean: 33.418335799999 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.591755891074055,
            "unit": "iter/sec",
            "range": "stddev: 0.0005039696216505472",
            "extra": "mean: 32.688545357142324 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 30.40082699977727,
            "unit": "iter/sec",
            "range": "stddev: 0.0015699068259927452",
            "extra": "mean: 32.89384199999975 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bb0bd93735bb2e0b2022850fd3b0494352ac5717",
          "message": "fix: preserve aspect ratio in braille sparsity display (#91)\n\n* fix: preserve aspect ratio in braille sparsity display\n\n`_render_braille` scaled rows and columns independently, so\nvery tall/slim matrices (e.g. 20000×20) rendered as nearly\nsquare grids instead of reflecting their true proportions.\n\nUse a single uniform scale factor matching Julia's SparseArrays:\n`s = min(2*max_width/n, 4*max_height/m)`, applied to both\ndimensions with floor, and clamp to minimum 8×4 dot space.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: add braille aspect ratio preservation test\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-23T19:43:43+01:00",
          "tree_id": "5aff008e36f35ad24a40c204a844c6234314cd05",
          "url": "https://github.com/adrhill/asdex/commit/bb0bd93735bb2e0b2022850fd3b0494352ac5717"
        },
        "date": 1774291460920,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 880.0435725981779,
            "unit": "iter/sec",
            "range": "stddev: 0.0033771972040575063",
            "extra": "mean: 1.1363073728811761 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3451.639696736743,
            "unit": "iter/sec",
            "range": "stddev: 0.000007030479726512234",
            "extra": "mean: 289.7173772063817 usec\nrounds: 1983"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 78.3435012490925,
            "unit": "iter/sec",
            "range": "stddev: 0.0004966240767595383",
            "extra": "mean: 12.764300600001377 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 81.35177294084346,
            "unit": "iter/sec",
            "range": "stddev: 0.00016103303157862715",
            "extra": "mean: 12.292295101265582 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 95.69082585067243,
            "unit": "iter/sec",
            "range": "stddev: 0.0004611425847287421",
            "extra": "mean: 10.450322600000561 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.681951060063223,
            "unit": "iter/sec",
            "range": "stddev: 0.016870228847686585",
            "extra": "mean: 48.351337700000485 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.048373791927,
            "unit": "iter/sec",
            "range": "stddev: 0.000029976305239847327",
            "extra": "mean: 4.047790255208221 msec\nrounds: 192"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 38.55588114369644,
            "unit": "iter/sec",
            "range": "stddev: 0.0007475740644988356",
            "extra": "mean: 25.93638039999746 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 38.390180099051534,
            "unit": "iter/sec",
            "range": "stddev: 0.0008323440812814409",
            "extra": "mean: 26.048327916667052 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 43.950056702605465,
            "unit": "iter/sec",
            "range": "stddev: 0.0006334193522990392",
            "extra": "mean: 22.753099199999838 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 106.19595935418629,
            "unit": "iter/sec",
            "range": "stddev: 0.011114983128969896",
            "extra": "mean: 9.416554133333696 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3414.679102836436,
            "unit": "iter/sec",
            "range": "stddev: 0.000007071229665374957",
            "extra": "mean: 292.85328720035227 usec\nrounds: 2211"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 33.07752870611924,
            "unit": "iter/sec",
            "range": "stddev: 0.00045306878350449755",
            "extra": "mean: 30.23200459999913 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 34.41328074970938,
            "unit": "iter/sec",
            "range": "stddev: 0.00025949263739523376",
            "extra": "mean: 29.058548857142746 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 33.366449498582355,
            "unit": "iter/sec",
            "range": "stddev: 0.0001890004911513693",
            "extra": "mean: 29.97022503225844 msec\nrounds: 31"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "31056ab70cd191135dac0206070e2581172568d2",
          "message": "test: cover all `ColoringMode` values in save/load roundtrip (#92)\n\nAdd `rev_over_fwd` and `rev_over_rev` to the\n`test_save_load_colored_roundtrip` parametrization.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-23T19:55:12+01:00",
          "tree_id": "68d4db919413bba1428f78d7dc2ef81c722a39de",
          "url": "https://github.com/adrhill/asdex/commit/31056ab70cd191135dac0206070e2581172568d2"
        },
        "date": 1774292148723,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 846.2182268104781,
            "unit": "iter/sec",
            "range": "stddev: 0.003741495486184225",
            "extra": "mean: 1.181728268568674 msec\nrounds: 175"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3474.4568816149194,
            "unit": "iter/sec",
            "range": "stddev: 0.000009554492097125842",
            "extra": "mean: 287.8147676235378 usec\nrounds: 2156"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 79.98523152692016,
            "unit": "iter/sec",
            "range": "stddev: 0.0005183257279151888",
            "extra": "mean: 12.50230799998917 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 82.11235535031366,
            "unit": "iter/sec",
            "range": "stddev: 0.00016158596435457206",
            "extra": "mean: 12.178435215184459 msec\nrounds: 79"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 97.42531584762432,
            "unit": "iter/sec",
            "range": "stddev: 0.0002875560687752124",
            "extra": "mean: 10.264272599988544 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.3873037042817,
            "unit": "iter/sec",
            "range": "stddev: 0.017490288072048877",
            "extra": "mean: 49.050135049981236 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.4858791807956,
            "unit": "iter/sec",
            "range": "stddev: 0.000030302718818796602",
            "extra": "mean: 4.141028880828763 msec\nrounds: 193"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 38.2104575450567,
            "unit": "iter/sec",
            "range": "stddev: 0.0012409200625355392",
            "extra": "mean: 26.17084599996815 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 38.450315786203426,
            "unit": "iter/sec",
            "range": "stddev: 0.0003487193094751078",
            "extra": "mean: 26.007588742842408 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 43.51096863223891,
            "unit": "iter/sec",
            "range": "stddev: 0.0008875611231794075",
            "extra": "mean: 22.98271059999024 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 119.15106481924373,
            "unit": "iter/sec",
            "range": "stddev: 0.008357045301183362",
            "extra": "mean: 8.392707203389534 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3479.3364957824224,
            "unit": "iter/sec",
            "range": "stddev: 0.000007381137482314868",
            "extra": "mean: 287.41112025588177 usec\nrounds: 2187"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 32.732271290274014,
            "unit": "iter/sec",
            "range": "stddev: 0.0003421579554796712",
            "extra": "mean: 30.550889400001324 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 34.26997371773318,
            "unit": "iter/sec",
            "range": "stddev: 0.00027717250781252983",
            "extra": "mean: 29.180063230761824 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 33.060002999593195,
            "unit": "iter/sec",
            "range": "stddev: 0.00030973277319565063",
            "extra": "mean: 30.248031133339737 msec\nrounds: 30"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "49f601de0e51a35c3b2d44fd94b8437839b0e585",
          "message": "test: add `@pytest.mark.fallback` to dynamic slice fallback tests\n\nAdd missing `fallback` marker and `TODO` comments to\n`test_dynamic_slice_dynamic_start` and\n`test_dynamic_update_slice_dynamic_start`, documenting the\nconservative union-over-windows pattern.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-23T19:58:38+01:00",
          "tree_id": "f4bdedc75afd849c55594ba95683e61337c5a567",
          "url": "https://github.com/adrhill/asdex/commit/49f601de0e51a35c3b2d44fd94b8437839b0e585"
        },
        "date": 1774292369928,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 795.7656765136944,
            "unit": "iter/sec",
            "range": "stddev: 0.0036950109394201366",
            "extra": "mean: 1.2566513353291016 msec\nrounds: 167"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3382.5832079229403,
            "unit": "iter/sec",
            "range": "stddev: 0.000007363371483670781",
            "extra": "mean: 295.63204761902824 usec\nrounds: 2100"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.38434988807911,
            "unit": "iter/sec",
            "range": "stddev: 0.0005538311472530124",
            "extra": "mean: 14.412471999997933 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 71.07600757515847,
            "unit": "iter/sec",
            "range": "stddev: 0.0001817886356566436",
            "extra": "mean: 14.069445289854839 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.70679959037497,
            "unit": "iter/sec",
            "range": "stddev: 0.0003448299483073588",
            "extra": "mean: 11.946460800001546 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.175348054298823,
            "unit": "iter/sec",
            "range": "stddev: 0.0004037373839313668",
            "extra": "mean: 41.36445100000046 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 255.92461899953153,
            "unit": "iter/sec",
            "range": "stddev: 0.00008567886008514875",
            "extra": "mean: 3.9074005615764165 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.56595458242858,
            "unit": "iter/sec",
            "range": "stddev: 0.0007700600452331395",
            "extra": "mean: 28.930200599995715 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 35.37361222725752,
            "unit": "iter/sec",
            "range": "stddev: 0.0005404886830734904",
            "extra": "mean: 28.269660264705426 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.06847381627634,
            "unit": "iter/sec",
            "range": "stddev: 0.0007864463650171801",
            "extra": "mean: 24.957276999998612 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 98.96918244814,
            "unit": "iter/sec",
            "range": "stddev: 0.011289113971602828",
            "extra": "mean: 10.10415540740676 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3372.7724489950806,
            "unit": "iter/sec",
            "range": "stddev: 0.000008463231294565925",
            "extra": "mean: 296.4919854874735 usec\nrounds: 2205"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.442604849249957,
            "unit": "iter/sec",
            "range": "stddev: 0.0007956635027333719",
            "extra": "mean: 32.84870019999744 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.974040108301413,
            "unit": "iter/sec",
            "range": "stddev: 0.0003937388572833447",
            "extra": "mean: 32.28510057142943 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 30.494498984814964,
            "unit": "iter/sec",
            "range": "stddev: 0.00024958191800220655",
            "extra": "mean: 32.792799793102354 msec\nrounds: 29"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "77d256984198939ca68eed72e76a62aa5b894176",
          "message": "asdex `v0.1.6`\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-23T20:04:48+01:00",
          "tree_id": "4dae63e6fd0fa3c8a9cdb0d65b02d086c5abb7e1",
          "url": "https://github.com/adrhill/asdex/commit/77d256984198939ca68eed72e76a62aa5b894176"
        },
        "date": 1774292774819,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 800.6581791496811,
            "unit": "iter/sec",
            "range": "stddev: 0.0037463330170009034",
            "extra": "mean: 1.24897244047644 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3395.6925131036787,
            "unit": "iter/sec",
            "range": "stddev: 0.0000073189751224680855",
            "extra": "mean: 294.4907397065806 usec\nrounds: 2113"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 67.37360623884237,
            "unit": "iter/sec",
            "range": "stddev: 0.0005826252250206556",
            "extra": "mean: 14.842607599999269 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 68.75302277593356,
            "unit": "iter/sec",
            "range": "stddev: 0.0003014457921239725",
            "extra": "mean: 14.544815044118204 msec\nrounds: 68"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.34808733392482,
            "unit": "iter/sec",
            "range": "stddev: 0.0004829625154509707",
            "extra": "mean: 11.997875800000202 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.881027436208523,
            "unit": "iter/sec",
            "range": "stddev: 0.00044644772277340844",
            "extra": "mean: 41.87424526315796 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 254.11096777243682,
            "unit": "iter/sec",
            "range": "stddev: 0.000047796444040760095",
            "extra": "mean: 3.9352886212118436 msec\nrounds: 198"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.14618042669686,
            "unit": "iter/sec",
            "range": "stddev: 0.0011822204622362158",
            "extra": "mean: 29.285852399999612 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 34.49683217471843,
            "unit": "iter/sec",
            "range": "stddev: 0.0010737810637588727",
            "extra": "mean: 28.988168969696485 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 37.91537276665452,
            "unit": "iter/sec",
            "range": "stddev: 0.0005726256295679272",
            "extra": "mean: 26.374526400000775 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 81.78803817897533,
            "unit": "iter/sec",
            "range": "stddev: 0.016608823441070817",
            "extra": "mean: 12.22672682051277 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3254.750888833834,
            "unit": "iter/sec",
            "range": "stddev: 0.000007911294966971354",
            "extra": "mean: 307.2431759464997 usec\nrounds: 2245"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.855141889908488,
            "unit": "iter/sec",
            "range": "stddev: 0.000379415084200992",
            "extra": "mean: 34.655868400000145 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.003440687421705,
            "unit": "iter/sec",
            "range": "stddev: 0.0005651486374320331",
            "extra": "mean: 33.329510785715605 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 29.50528704395565,
            "unit": "iter/sec",
            "range": "stddev: 0.0005247583391165198",
            "extra": "mean: 33.892230857142486 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": false,
          "id": "78cc5e99edf40161271b79077e62440cdb25a24c",
          "message": "docs: update make-release skill to link commits without PRs\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-23T20:07:04+01:00",
          "tree_id": "7ed826c5c670d914dd4a1ef3c0dc227e8eb2fec0",
          "url": "https://github.com/adrhill/asdex/commit/78cc5e99edf40161271b79077e62440cdb25a24c"
        },
        "date": 1774293597644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 765.8268163064411,
            "unit": "iter/sec",
            "range": "stddev: 0.004402540964637458",
            "extra": "mean: 1.305778250000397 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3383.672776696365,
            "unit": "iter/sec",
            "range": "stddev: 0.000008329785218975783",
            "extra": "mean: 295.5368518158975 usec\nrounds: 2065"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.17883132919164,
            "unit": "iter/sec",
            "range": "stddev: 0.00045718888752217877",
            "extra": "mean: 14.667309200001455 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 68.76048747847022,
            "unit": "iter/sec",
            "range": "stddev: 0.0003118436972945707",
            "extra": "mean: 14.543236045455792 msec\nrounds: 66"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 85.33667725670767,
            "unit": "iter/sec",
            "range": "stddev: 0.0006439136465283557",
            "extra": "mean: 11.718290800001796 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.118815459368896,
            "unit": "iter/sec",
            "range": "stddev: 0.00042176541522163604",
            "extra": "mean: 41.46140600000123 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 253.42759000739997,
            "unit": "iter/sec",
            "range": "stddev: 0.000028327209270199597",
            "extra": "mean: 3.9459002864321144 msec\nrounds: 199"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 33.8556891499575,
            "unit": "iter/sec",
            "range": "stddev: 0.0004276385594126856",
            "extra": "mean: 29.537133199997356 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 34.43869174159875,
            "unit": "iter/sec",
            "range": "stddev: 0.00041906724510855527",
            "extra": "mean: 29.0371076666682 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 38.19705325012517,
            "unit": "iter/sec",
            "range": "stddev: 0.0003201505049350613",
            "extra": "mean: 26.180030000003285 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.75742396919739,
            "unit": "iter/sec",
            "range": "stddev: 0.013223844431551595",
            "extra": "mean: 10.44305452830136 msec\nrounds: 53"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3336.828883696396,
            "unit": "iter/sec",
            "range": "stddev: 0.00000771857544506901",
            "extra": "mean: 299.68573003127534 usec\nrounds: 2241"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.130830065176767,
            "unit": "iter/sec",
            "range": "stddev: 0.00031234702313999506",
            "extra": "mean: 34.327892399997495 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.17617809122266,
            "unit": "iter/sec",
            "range": "stddev: 0.00046594261868485064",
            "extra": "mean: 33.13872276923199 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.910171320301725,
            "unit": "iter/sec",
            "range": "stddev: 0.00038982704695880314",
            "extra": "mean: 34.58990225000034 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bc83e433249781e691454f2e97ba3a3f573d6264",
          "message": "ci(prek): add typos and standard hooks to pre-commit config  (#93)\n\n* ci(prek): add `typos` and standard hooks to pre-commit config\n\nAdd `typos` spell checker and standard `pre-commit-hooks`\n(trailing-whitespace, end-of-file-fixer, check-merge-conflict,\ncheck-added-large-files, check-toml, check-yaml).\n\nFix trailing whitespace and missing final newlines caught by the\nnew hooks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-23T20:34:32+01:00",
          "tree_id": "1e2adf241d11f300e911bc5387831ce52acddc1c",
          "url": "https://github.com/adrhill/asdex/commit/bc83e433249781e691454f2e97ba3a3f573d6264"
        },
        "date": 1774294508236,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 745.6772656153832,
            "unit": "iter/sec",
            "range": "stddev: 0.004661209915188513",
            "extra": "mean: 1.341062744047497 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3283.7824458476293,
            "unit": "iter/sec",
            "range": "stddev: 0.000038409624891587766",
            "extra": "mean: 304.5268730468148 usec\nrounds: 2048"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.3158056609261,
            "unit": "iter/sec",
            "range": "stddev: 0.0006784177290028293",
            "extra": "mean: 14.637901000001818 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 68.98242228446952,
            "unit": "iter/sec",
            "range": "stddev: 0.0013017413311289406",
            "extra": "mean: 14.496446585714295 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.20849588921688,
            "unit": "iter/sec",
            "range": "stddev: 0.0016072761623340808",
            "extra": "mean: 12.467507200000227 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.743892527474834,
            "unit": "iter/sec",
            "range": "stddev: 0.012157840105679973",
            "extra": "mean: 43.96784757894853 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 254.30162721542524,
            "unit": "iter/sec",
            "range": "stddev: 0.00002825931132444777",
            "extra": "mean: 3.9323381881189263 msec\nrounds: 202"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.17488789731484,
            "unit": "iter/sec",
            "range": "stddev: 0.0007296533948644935",
            "extra": "mean: 29.261251799997012 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 33.26482850782315,
            "unit": "iter/sec",
            "range": "stddev: 0.002443887258088065",
            "extra": "mean: 30.061781312500145 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 39.036360582017714,
            "unit": "iter/sec",
            "range": "stddev: 0.0011631702139563616",
            "extra": "mean: 25.61714219999942 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 126.7506830752634,
            "unit": "iter/sec",
            "range": "stddev: 0.0002462838571125196",
            "extra": "mean: 7.889503833334052 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3344.892100176205,
            "unit": "iter/sec",
            "range": "stddev: 0.00000856876390658686",
            "extra": "mean: 298.9633058559112 usec\nrounds: 2220"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.56072238063846,
            "unit": "iter/sec",
            "range": "stddev: 0.00063113964662437",
            "extra": "mean: 35.0131200000007 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 30.41950687850049,
            "unit": "iter/sec",
            "range": "stddev: 0.0004934335284945046",
            "extra": "mean: 32.873642692307 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 27.811307926357717,
            "unit": "iter/sec",
            "range": "stddev: 0.003663126390920845",
            "extra": "mean: 35.95659731818172 msec\nrounds: 22"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "45d3cbda58aeb9915c7ba2afe0fd3007db3f14cb",
          "message": "docs: add coloring algorithm references and subsections to API docs (#94)\n\n* docs: add `color_rows`, `color_cols`, `color_symmetric` to reference pages and add subsections\n\nAdd the three coloring algorithms exported in 669b5c5 to\n`coloring.md` and `index.md`.\nAdd subsection headings to coloring, jacobian, hessian, and index\nreference pages for better scannability.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: use monochrome SVG as favicon\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add `VerificationError` directive to full API reference\n\nFixes cross-reference warnings in `how-to/jacobians.md` and\n`how-to/hessians.md`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-31T15:55:34+02:00",
          "tree_id": "9c3dcb857ac2352aaada85592d2c2a643f0a0983",
          "url": "https://github.com/adrhill/asdex/commit/45d3cbda58aeb9915c7ba2afe0fd3007db3f14cb"
        },
        "date": 1774965370847,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 846.3159677054459,
            "unit": "iter/sec",
            "range": "stddev: 0.0029253546064354",
            "extra": "mean: 1.1815917909610358 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3261.812701885005,
            "unit": "iter/sec",
            "range": "stddev: 0.000013532179450289222",
            "extra": "mean: 306.57799554894706 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.68538844999252,
            "unit": "iter/sec",
            "range": "stddev: 0.0005790454822411655",
            "extra": "mean: 14.350210599997126 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 71.88728596078082,
            "unit": "iter/sec",
            "range": "stddev: 0.00020962054522397848",
            "extra": "mean: 13.910665657145062 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 85.51752357530177,
            "unit": "iter/sec",
            "range": "stddev: 0.00026719974504402753",
            "extra": "mean: 11.693509800005586 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.937610200078538,
            "unit": "iter/sec",
            "range": "stddev: 0.000308264004156553",
            "extra": "mean: 41.775264600002515 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.2392928163343,
            "unit": "iter/sec",
            "range": "stddev: 0.00003786110017529791",
            "extra": "mean: 4.179915381908887 msec\nrounds: 199"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 35.41306396287173,
            "unit": "iter/sec",
            "range": "stddev: 0.0006953156488953515",
            "extra": "mean: 28.238166599999204 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 35.95783143039504,
            "unit": "iter/sec",
            "range": "stddev: 0.00012919455304598605",
            "extra": "mean: 27.810353411766187 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.535251128489264,
            "unit": "iter/sec",
            "range": "stddev: 0.0008993081248467207",
            "extra": "mean: 24.669885399998748 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 103.02957622575177,
            "unit": "iter/sec",
            "range": "stddev: 0.009270850184139205",
            "extra": "mean: 9.705950821430775 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3254.1561376707273,
            "unit": "iter/sec",
            "range": "stddev: 0.000007810690559376888",
            "extra": "mean: 307.29932974752217 usec\nrounds: 2138"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 31.294161930139804,
            "unit": "iter/sec",
            "range": "stddev: 0.00043118756059061573",
            "extra": "mean: 31.954841999998962 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 31.80518434046176,
            "unit": "iter/sec",
            "range": "stddev: 0.00023123013911945404",
            "extra": "mean: 31.44141500000127 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 31.539215215274872,
            "unit": "iter/sec",
            "range": "stddev: 0.00022814393179920183",
            "extra": "mean: 31.706559379311578 msec\nrounds: 29"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "346748459d5b0b08b205300cdd8f6f41747c5e68",
          "message": "chore(ty): upgrade `ty` and pin linter versions (#95)\n\n* build: upgrade `ty` to v0.0.26 and migrate to `ty: ignore` syntax\n\nPin `ty>=0.0.26` in `pyproject.toml` so CI uses a compatible version.\nReplace `# type: ignore[...]` comments with `# ty: ignore[...]`\nusing ty's own rule names (e.g. `invalid-argument-type` instead of\n`arg-type`).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* build: pin linter versions to prevent local/CI drift\n\nUse exact (`==`) pins for `ruff`, `ty`, and `prek` so local and CI\nalways resolve the same versions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-31T16:29:35+02:00",
          "tree_id": "819464870cd8c9a075d05934c151b27385b43f28",
          "url": "https://github.com/adrhill/asdex/commit/346748459d5b0b08b205300cdd8f6f41747c5e68"
        },
        "date": 1774967418048,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 761.1099465133599,
            "unit": "iter/sec",
            "range": "stddev: 0.00453396439308559",
            "extra": "mean: 1.3138706235294837 msec\nrounds: 170"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3241.9206046395366,
            "unit": "iter/sec",
            "range": "stddev: 0.00000783049956693432",
            "extra": "mean: 308.4591271510143 usec\nrounds: 2092"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.34804123945295,
            "unit": "iter/sec",
            "range": "stddev: 0.00051962448718492",
            "extra": "mean: 14.630997200001161 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 68.95870669487493,
            "unit": "iter/sec",
            "range": "stddev: 0.00028274544864588964",
            "extra": "mean: 14.501432058822543 msec\nrounds: 68"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.09814619923283,
            "unit": "iter/sec",
            "range": "stddev: 0.0004550743230900491",
            "extra": "mean: 12.180542999999489 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.153257718212007,
            "unit": "iter/sec",
            "range": "stddev: 0.019283221590199053",
            "extra": "mean: 49.6197693684195 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.03883433553506,
            "unit": "iter/sec",
            "range": "stddev: 0.00006639724779463679",
            "extra": "mean: 4.165992568528155 msec\nrounds: 197"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 23.57924097017912,
            "unit": "iter/sec",
            "range": "stddev: 0.028650154437046924",
            "extra": "mean: 42.410186200001476 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 34.556245238525,
            "unit": "iter/sec",
            "range": "stddev: 0.0006157552279228974",
            "extra": "mean: 28.93832918181605 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 38.59358016171608,
            "unit": "iter/sec",
            "range": "stddev: 0.0014896711942890036",
            "extra": "mean: 25.91104519999874 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 94.35637287581064,
            "unit": "iter/sec",
            "range": "stddev: 0.013466637901264292",
            "extra": "mean: 10.59811827777837 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3295.7299451260465,
            "unit": "iter/sec",
            "range": "stddev: 0.000008030610352119678",
            "extra": "mean: 303.4229189436074 usec\nrounds: 2196"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.595005112087055,
            "unit": "iter/sec",
            "range": "stddev: 0.00041004078452157013",
            "extra": "mean: 32.685073800001874 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 31.56669966787008,
            "unit": "iter/sec",
            "range": "stddev: 0.00028061155126218694",
            "extra": "mean: 31.678953153846564 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 31.313386054233078,
            "unit": "iter/sec",
            "range": "stddev: 0.0005796313979944463",
            "extra": "mean: 31.935224068967003 msec\nrounds: 29"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "45ec844dd396c78e92effb509c01f96a72ed22a5",
          "message": "docs: fix incorrect API references in amortization section (#96)\n\nThe amortization section incorrectly linked to `jacobian` and `hessian`\n(one-call APIs that don't accept pre-computed colorings) instead of\n`jacobian_from_coloring` and `hessian_from_coloring`.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-01T18:28:01+02:00",
          "tree_id": "8705728deba8e3103d9099310978a626ec77dea6",
          "url": "https://github.com/adrhill/asdex/commit/45ec844dd396c78e92effb509c01f96a72ed22a5"
        },
        "date": 1775060921997,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 755.3398329860769,
            "unit": "iter/sec",
            "range": "stddev: 0.004467470401739168",
            "extra": "mean: 1.3239074074072203 msec\nrounds: 162"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3260.2075128366987,
            "unit": "iter/sec",
            "range": "stddev: 0.000010334482831185008",
            "extra": "mean: 306.7289416586561 usec\nrounds: 2074"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 66.35403036305435,
            "unit": "iter/sec",
            "range": "stddev: 0.0005484947935051813",
            "extra": "mean: 15.070674599998313 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 68.07333033411523,
            "unit": "iter/sec",
            "range": "stddev: 0.0004645634774498939",
            "extra": "mean: 14.690040799999556 msec\nrounds: 65"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.67295628470157,
            "unit": "iter/sec",
            "range": "stddev: 0.0003253299294585269",
            "extra": "mean: 11.95129279999918 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.763348916865755,
            "unit": "iter/sec",
            "range": "stddev: 0.022146543501898555",
            "extra": "mean: 50.598711999999885 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.69758330262476,
            "unit": "iter/sec",
            "range": "stddev: 0.00005874367940585615",
            "extra": "mean: 4.189401443299002 msec\nrounds: 194"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 33.59616035127558,
            "unit": "iter/sec",
            "range": "stddev: 0.001382783505443521",
            "extra": "mean: 29.765306199999486 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 33.05273451929682,
            "unit": "iter/sec",
            "range": "stddev: 0.001002046846613473",
            "extra": "mean: 30.254682843750214 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 37.83515534577256,
            "unit": "iter/sec",
            "range": "stddev: 0.000645453099413771",
            "extra": "mean: 26.430445200001884 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 103.3456471924321,
            "unit": "iter/sec",
            "range": "stddev: 0.010641356514698705",
            "extra": "mean: 9.676266269231212 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3209.546601949784,
            "unit": "iter/sec",
            "range": "stddev: 0.000009564984019153882",
            "extra": "mean: 311.57048767963204 usec\nrounds: 1948"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 28.316510400116297,
            "unit": "iter/sec",
            "range": "stddev: 0.00046692648200685164",
            "extra": "mean: 35.31508599999995 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 29.66505667623558,
            "unit": "iter/sec",
            "range": "stddev: 0.00031897544154096946",
            "extra": "mean: 33.70969457142792 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.996984230944605,
            "unit": "iter/sec",
            "range": "stddev: 0.0022537045313480207",
            "extra": "mean: 34.48634492592625 msec\nrounds: 27"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e57175e7bb09b88689f6b758fb9f6c5c02293f18",
          "message": "feat: add `spy` matplotlib visualization (#97)\n\n* feat: add `spy` matplotlib visualization as optional `asdex[matplotlib]` dependency\n\nAdd `_plotting.py` with `spy()` that accepts `SparsityPattern` (black markers)\nor `ColoredPattern` (colored by color assignment). `compressed=True` shows\noriginal and compressed patterns side by side.\n\nmatplotlib is an optional dependency (`pip install asdex[matplotlib]`) with a\nlazy import and clear error message when missing.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add visualization how-to guide with `mkdocs-matplotlib`\n\nAdd how-to page covering `print` output (dot/braille) and `spy` matplotlib\nplots for sparsity patterns, colored patterns, and compressed views.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: use grid-based rendering in `spy` and improve docs\n\nSwitch `spy` from `scatter` to `imshow` for proper grid display.\nReplace `markersize` with configurable `cmap` (default `tab10`).\n`compressed` now controls what a single plot shows, letting users\ncompose their own subplots. Remove \"Larger Patterns\" section.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix: correct colormap handling and add Brusselator example to visualization docs\n\nHandle discrete vs continuous colormaps correctly: discrete colormaps\n(N <= 20) map index i to entry i, continuous colormaps spread colors\nevenly for maximum contrast. Cycle indices when `num_colors > cmap.N`.\n\nAdd \"Larger Example\" section to visualization docs using an N=8\nBrusselator PDE.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: add smoke tests for `spy` matplotlib visualization\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add row coloring and symmetric coloring examples to visualization guide\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: use 2D grid Hessian for symmetric coloring example\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* ci: install `asdex[matplotlib]` in CI workflows\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-01T20:14:35+02:00",
          "tree_id": "0ca7d2f71d5fe44afcb91e48bcc557a1c452b664",
          "url": "https://github.com/adrhill/asdex/commit/e57175e7bb09b88689f6b758fb9f6c5c02293f18"
        },
        "date": 1775067315703,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 808.9641460630016,
            "unit": "iter/sec",
            "range": "stddev: 0.0033967480116471765",
            "extra": "mean: 1.2361487278103924 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3305.631626308491,
            "unit": "iter/sec",
            "range": "stddev: 0.000010222932246770509",
            "extra": "mean: 302.5140466473372 usec\nrounds: 2058"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 68.5767071752779,
            "unit": "iter/sec",
            "range": "stddev: 0.00040107408911354187",
            "extra": "mean: 14.582210800003281 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 70.18819368117545,
            "unit": "iter/sec",
            "range": "stddev: 0.00013537182788806205",
            "extra": "mean: 14.247410391303475 msec\nrounds: 69"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.28637231226787,
            "unit": "iter/sec",
            "range": "stddev: 0.00033133930977655347",
            "extra": "mean: 12.006766200005359 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.705027050453065,
            "unit": "iter/sec",
            "range": "stddev: 0.01656553094548225",
            "extra": "mean: 48.29744957894745 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.07094053814257,
            "unit": "iter/sec",
            "range": "stddev: 0.00006286498725348445",
            "extra": "mean: 4.131020426396172 msec\nrounds: 197"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 22.736511191858924,
            "unit": "iter/sec",
            "range": "stddev: 0.03065210064607115",
            "extra": "mean: 43.982121599995594 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 34.335487351532464,
            "unit": "iter/sec",
            "range": "stddev: 0.0006239110223890793",
            "extra": "mean: 29.124386375001254 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 38.91565360528865,
            "unit": "iter/sec",
            "range": "stddev: 0.0010165500937639321",
            "extra": "mean: 25.696600400002012 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 91.31913025367948,
            "unit": "iter/sec",
            "range": "stddev: 0.014702161480136155",
            "extra": "mean: 10.950608018517649 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3256.6806326700994,
            "unit": "iter/sec",
            "range": "stddev: 0.000010173134076495378",
            "extra": "mean: 307.0611192169974 usec\nrounds: 2248"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.28182402807005,
            "unit": "iter/sec",
            "range": "stddev: 0.0008566272181126394",
            "extra": "mean: 34.150877999996965 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 29.630297549840478,
            "unit": "iter/sec",
            "range": "stddev: 0.0008335629384184254",
            "extra": "mean: 33.74923921428469 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 28.892936938405118,
            "unit": "iter/sec",
            "range": "stddev: 0.0007863197237824631",
            "extra": "mean: 34.610534821428224 msec\nrounds: 28"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "ffd967334986adaa9951a35833df4c765214634c",
          "message": "asdex `v0.1.7`",
          "timestamp": "2026-04-01T20:29:25+02:00",
          "tree_id": "cbaa9815c90ba8854d7423e2249c29e70897d02b",
          "url": "https://github.com/adrhill/asdex/commit/ffd967334986adaa9951a35833df4c765214634c"
        },
        "date": 1775068212602,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 753.4225372840367,
            "unit": "iter/sec",
            "range": "stddev: 0.004444033781402538",
            "extra": "mean: 1.3272764624281537 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3290.7151041929287,
            "unit": "iter/sec",
            "range": "stddev: 0.000008958312816390104",
            "extra": "mean: 303.88531621161326 usec\nrounds: 2122"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 67.14429898044672,
            "unit": "iter/sec",
            "range": "stddev: 0.0005660259559933582",
            "extra": "mean: 14.893297199978406 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 69.57208129100594,
            "unit": "iter/sec",
            "range": "stddev: 0.0006790703959468499",
            "extra": "mean: 14.373581779409506 msec\nrounds: 68"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.27297707959808,
            "unit": "iter/sec",
            "range": "stddev: 0.00027673025109995874",
            "extra": "mean: 12.008697599992502 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.142462506280445,
            "unit": "iter/sec",
            "range": "stddev: 0.018326161398124062",
            "extra": "mean: 49.64636273684008 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.76519324564288,
            "unit": "iter/sec",
            "range": "stddev: 0.00006770762072554214",
            "extra": "mean: 4.119206656566068 msec\nrounds: 198"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 33.3066833237341,
            "unit": "iter/sec",
            "range": "stddev: 0.0011548392245360902",
            "extra": "mean: 30.02400420000413 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 33.71842976348574,
            "unit": "iter/sec",
            "range": "stddev: 0.0009578189996615067",
            "extra": "mean: 29.65737156250725 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 39.47778221209402,
            "unit": "iter/sec",
            "range": "stddev: 0.0009640426529322249",
            "extra": "mean: 25.33070360000238 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.85389986227999,
            "unit": "iter/sec",
            "range": "stddev: 0.012329662604814691",
            "extra": "mean: 10.432543709090293 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3219.6572486999385,
            "unit": "iter/sec",
            "range": "stddev: 0.000008158208132328708",
            "extra": "mean: 310.59206702942953 usec\nrounds: 2208"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.94687880362063,
            "unit": "iter/sec",
            "range": "stddev: 0.00040616214836125336",
            "extra": "mean: 32.31343640002251 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 31.553130502136344,
            "unit": "iter/sec",
            "range": "stddev: 0.0003121110756705737",
            "extra": "mean: 31.692576428582697 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 31.242753936754283,
            "unit": "iter/sec",
            "range": "stddev: 0.00027033709635396896",
            "extra": "mean: 32.00742168966065 msec\nrounds: 29"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "a69d93378bdbf720f8166250c581f63bbf7f8c96",
          "message": "fix(ci): install matplotlib extra in release workflow",
          "timestamp": "2026-04-01T21:49:07+02:00",
          "tree_id": "70fdfa4b6fb9a982b4a1ed268d01b57ee906f26a",
          "url": "https://github.com/adrhill/asdex/commit/a69d93378bdbf720f8166250c581f63bbf7f8c96"
        },
        "date": 1775072998376,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 761.1157342267517,
            "unit": "iter/sec",
            "range": "stddev: 0.00440227922453753",
            "extra": "mean: 1.3138606325304003 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3286.8644869528503,
            "unit": "iter/sec",
            "range": "stddev: 0.000008062710602239613",
            "extra": "mean: 304.24132299018777 usec\nrounds: 2127"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 69.55459380408911,
            "unit": "iter/sec",
            "range": "stddev: 0.0005927176426799362",
            "extra": "mean: 14.377195600000903 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 71.61499886284561,
            "unit": "iter/sec",
            "range": "stddev: 0.00015284569230554",
            "extra": "mean: 13.963555342857196 msec\nrounds: 70"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.85627644688972,
            "unit": "iter/sec",
            "range": "stddev: 0.0003381264771885343",
            "extra": "mean: 11.925165800002446 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.145886103969183,
            "unit": "iter/sec",
            "range": "stddev: 0.012137478927052574",
            "extra": "mean: 45.15511347368355 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.38602499090536,
            "unit": "iter/sec",
            "range": "stddev: 0.00003416370946409786",
            "extra": "mean: 4.159975606060433 msec\nrounds: 198"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 34.72604956342191,
            "unit": "iter/sec",
            "range": "stddev: 0.00105368544376933",
            "extra": "mean: 28.796825799999226 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 34.922451294607704,
            "unit": "iter/sec",
            "range": "stddev: 0.0007050347268620374",
            "extra": "mean: 28.634874212120607 msec\nrounds: 33"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 40.22449224764013,
            "unit": "iter/sec",
            "range": "stddev: 0.0007911404999120819",
            "extra": "mean: 24.86047540000129 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.77417455470592,
            "unit": "iter/sec",
            "range": "stddev: 0.00043846855268923695",
            "extra": "mean: 8.145035416665891 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3235.433446885785,
            "unit": "iter/sec",
            "range": "stddev: 0.00000883788696940435",
            "extra": "mean: 309.07759854016905 usec\nrounds: 2192"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 31.558400524323687,
            "unit": "iter/sec",
            "range": "stddev: 0.000419563840378486",
            "extra": "mean: 31.687284000000204 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 31.764105305868743,
            "unit": "iter/sec",
            "range": "stddev: 0.00037100282862518637",
            "extra": "mean: 31.482076714285412 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 31.437860439064238,
            "unit": "iter/sec",
            "range": "stddev: 0.00047005175116314745",
            "extra": "mean: 31.808780433334267 msec\nrounds: 30"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fdb48b31170dc6f49a5e4e685ff095894f6db92a",
          "message": "bench: `jit` materialization benchmarks to measure XLA compute time (#98)\n\nPreviously, materialization and end-to-end benchmarks measured Python\ntracing overhead instead of actual XLA execution time. Wrap all\ndecompression benchmarks in `jax.jit` with a warmup call and\n`block_until_ready()` to ensure only compiled execution is timed.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-01T21:51:11+02:00",
          "tree_id": "46b6cbe771ff5b81855d1b5b5c3479720e886b13",
          "url": "https://github.com/adrhill/asdex/commit/fdb48b31170dc6f49a5e4e685ff095894f6db92a"
        },
        "date": 1775073104438,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 850.746411662066,
            "unit": "iter/sec",
            "range": "stddev: 0.002921967830144586",
            "extra": "mean: 1.1754383989070771 msec\nrounds: 183"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3215.651625780416,
            "unit": "iter/sec",
            "range": "stddev: 0.000007536353099699372",
            "extra": "mean: 310.9789605263309 usec\nrounds: 2052"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 76342.38394354258,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024780604971764202",
            "extra": "mean: 13.098883586600193 usec\nrounds: 27703"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 44237.325213044554,
            "unit": "iter/sec",
            "range": "stddev: 0.000003750471317387737",
            "extra": "mean: 22.605345038020594 usec\nrounds: 12616"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 71896.5025670516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024218171083421882",
            "extra": "mean: 13.908882411455092 usec\nrounds: 21116"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.871016961392147,
            "unit": "iter/sec",
            "range": "stddev: 0.014336400172985908",
            "extra": "mean: 47.91333368421054 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 224.34900089536876,
            "unit": "iter/sec",
            "range": "stddev: 0.000020151416768362904",
            "extra": "mean: 4.45734100000016 msec\nrounds: 193"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1743.563086643008,
            "unit": "iter/sec",
            "range": "stddev: 0.00009842757089163145",
            "extra": "mean: 573.5381803278268 usec\nrounds: 1342"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1695.2964307507257,
            "unit": "iter/sec",
            "range": "stddev: 0.00008625472184564576",
            "extra": "mean: 589.8673422896145 usec\nrounds: 856"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4416.990256886017,
            "unit": "iter/sec",
            "range": "stddev: 0.000007883016162152885",
            "extra": "mean: 226.39850709224814 usec\nrounds: 3666"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 113.73948022513969,
            "unit": "iter/sec",
            "range": "stddev: 0.006595468121956088",
            "extra": "mean: 8.792021890908654 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3212.794871091443,
            "unit": "iter/sec",
            "range": "stddev: 0.000010329681362411453",
            "extra": "mean: 311.2554769674052 usec\nrounds: 2084"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 44467.32834305811,
            "unit": "iter/sec",
            "range": "stddev: 0.000006622579209196752",
            "extra": "mean: 22.488420988217797 usec\nrounds: 14713"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 41367.92012951721,
            "unit": "iter/sec",
            "range": "stddev: 0.0000065918752925728874",
            "extra": "mean: 24.173320700415655 usec\nrounds: 12850"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 45114.62814219321,
            "unit": "iter/sec",
            "range": "stddev: 0.000006226248621753128",
            "extra": "mean: 22.16575955914298 usec\nrounds: 16058"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0b883954fe5b5fda36b97cc623364a23907f7424",
          "message": "perf: use `lax.gather` for decompression (#99)\n\n* perf: use `lax.gather` with `unique_indices` for decompression\n\nReplace NumPy-style fancy indexing in `_decompress` with an explicit\n`jax.lax.gather` using `PROMISE_IN_BOUNDS` and `unique_indices=True`.\nThis eliminates 6 negative-index-normalization equations from the jaxpr\nand tells XLA that each gather index is unique, enabling better\nparallelization.\n\nAdd `_gather_indices` cached property on `ColoredPattern` that\nprecomputes the stacked `(nnz, 2)` index array, so it appears as a\nsingle closed-over constant in the trace.\n\nSplit `_decompress` into `_decompress_data` (pure gather, returns\n`jax.Array`) and `_decompress` (gather + BCOO wrap).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* claude: plan `jax.Array` return type as follow-up\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-01T22:48:25+02:00",
          "tree_id": "af56b0112d24486f9f8929e2fa5a1d00a323944f",
          "url": "https://github.com/adrhill/asdex/commit/0b883954fe5b5fda36b97cc623364a23907f7424"
        },
        "date": 1775076543135,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 779.0996886639059,
            "unit": "iter/sec",
            "range": "stddev: 0.0040722386397726105",
            "extra": "mean: 1.283532793749309 msec\nrounds: 160"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3328.2064846720473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000092320970744698",
            "extra": "mean: 300.46212715631356 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 77117.02298148717,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023712661486594106",
            "extra": "mean: 12.967305548608401 usec\nrounds: 24006"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 46350.93096777677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038510936462268866",
            "extra": "mean: 21.574539693608344 usec\nrounds: 11488"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 77604.97144736558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025024706663969136",
            "extra": "mean: 12.88577241057598 usec\nrounds: 23828"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.11607879512105,
            "unit": "iter/sec",
            "range": "stddev: 0.018980610571319882",
            "extra": "mean: 49.71147757894743 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.5761607419954,
            "unit": "iter/sec",
            "range": "stddev: 0.00010512751096039963",
            "extra": "mean: 4.1740380048786285 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1909.029080578461,
            "unit": "iter/sec",
            "range": "stddev: 0.00007273364315873384",
            "extra": "mean: 523.8264886446815 usec\nrounds: 1365"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1967.3275367080973,
            "unit": "iter/sec",
            "range": "stddev: 0.000022553262880812553",
            "extra": "mean: 508.303768102228 usec\nrounds: 1091"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4126.897154764006,
            "unit": "iter/sec",
            "range": "stddev: 0.000013905644815038182",
            "extra": "mean: 242.31279881681093 usec\nrounds: 3718"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 94.35054827066101,
            "unit": "iter/sec",
            "range": "stddev: 0.01281782027600937",
            "extra": "mean: 10.5987725384629 msec\nrounds: 52"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3272.380117996644,
            "unit": "iter/sec",
            "range": "stddev: 0.000008076082011081293",
            "extra": "mean: 305.58797081685043 usec\nrounds: 2193"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 45114.65922688987,
            "unit": "iter/sec",
            "range": "stddev: 0.000005742393547080252",
            "extra": "mean: 22.165744286592464 usec\nrounds: 9495"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 40714.34794547352,
            "unit": "iter/sec",
            "range": "stddev: 0.000006067913284362896",
            "extra": "mean: 24.561365967084747 usec\nrounds: 12247"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 45154.36661990458,
            "unit": "iter/sec",
            "range": "stddev: 0.000005747363067776892",
            "extra": "mean: 22.14625239719758 usec\nrounds: 13974"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "885c6cb2f31e8ae7eac426ceab8453cfd46fabbc",
          "message": "feat: add `output_format` kwarg for optional dense decompression (#100)\n\n* feat: add `output_format` parameter for dense decompression\n\nSupport direct decompression to dense arrays via `output_format=\"dense\"`,\nusing `jnp.zeros` + index assignment instead of BCOO wrapping.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\n\n* docs: improve `output_format` docstrings and extract `compressed_jacobian` variable\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\n\n* refactor: improve `_decompress` signature and docstrings\n\nSwap argument order in `_decompress` to lead with data,\nsplit long Returns docstrings into two sentences for readability.\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\n\n* test: add tests for `output_format=\"dense\"`\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\n\n* claude: remove completed dense output task from PLAN.md\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-02T00:56:15+02:00",
          "tree_id": "d171bf0e0f468d0b80b9e7be207223a1523ae98a",
          "url": "https://github.com/adrhill/asdex/commit/885c6cb2f31e8ae7eac426ceab8453cfd46fabbc"
        },
        "date": 1775084207958,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 769.1271829823123,
            "unit": "iter/sec",
            "range": "stddev: 0.004101115018563902",
            "extra": "mean: 1.300175084337121 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3259.664240784598,
            "unit": "iter/sec",
            "range": "stddev: 0.000011245310836596363",
            "extra": "mean: 306.7800626482012 usec\nrounds: 2107"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 72835.49300677261,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033422316252803368",
            "extra": "mean: 13.729570003830618 usec\nrounds: 20756"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 43417.40011944101,
            "unit": "iter/sec",
            "range": "stddev: 0.000004669508819719503",
            "extra": "mean: 23.03224046693275 usec\nrounds: 11565"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 77009.81618191412,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026111758762434354",
            "extra": "mean: 12.98535757620535 usec\nrounds: 25712"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.5492750589099,
            "unit": "iter/sec",
            "range": "stddev: 0.013438116792929961",
            "extra": "mean: 46.40527336842052 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 237.3598240580298,
            "unit": "iter/sec",
            "range": "stddev: 0.000030772514095439446",
            "extra": "mean: 4.213012897058432 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1937.2652562654112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000310240472509349",
            "extra": "mean: 516.1915730258661 usec\nrounds: 1431"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1929.479439415484,
            "unit": "iter/sec",
            "range": "stddev: 0.000032090541366257344",
            "extra": "mean: 518.2745042895817 usec\nrounds: 1049"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4245.477699783633,
            "unit": "iter/sec",
            "range": "stddev: 0.000012998985600665972",
            "extra": "mean: 235.5447539038926 usec\nrounds: 3202"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.08901960096468,
            "unit": "iter/sec",
            "range": "stddev: 0.011404621031290342",
            "extra": "mean: 10.40701637036955 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3236.8888710499914,
            "unit": "iter/sec",
            "range": "stddev: 0.000015703538938388687",
            "extra": "mean: 308.9386258959262 usec\nrounds: 2093"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 43998.661968206754,
            "unit": "iter/sec",
            "range": "stddev: 0.000006635068646407896",
            "extra": "mean: 22.727963880415174 usec\nrounds: 13483"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 40282.404421765845,
            "unit": "iter/sec",
            "range": "stddev: 0.000006932515291099085",
            "extra": "mean: 24.82473462928813 usec\nrounds: 12963"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 44486.8289565102,
            "unit": "iter/sec",
            "range": "stddev: 0.000006386489495182706",
            "extra": "mean: 22.478563283923613 usec\nrounds: 12278"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "35c5a6be26ebf1f0c6556069f95284bf3a40bd40",
          "message": "docs: use `jax.jit` in all examples (#101)\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-02T01:10:06+02:00",
          "tree_id": "55441630f155ccc7aed4d70bc91a00507081d569",
          "url": "https://github.com/adrhill/asdex/commit/35c5a6be26ebf1f0c6556069f95284bf3a40bd40"
        },
        "date": 1775085038602,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 746.285848894434,
            "unit": "iter/sec",
            "range": "stddev: 0.004462983154869087",
            "extra": "mean: 1.3399691304363124 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3248.646458002467,
            "unit": "iter/sec",
            "range": "stddev: 0.00003995096273544997",
            "extra": "mean: 307.82050707200733 usec\nrounds: 1909"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 74414.53998399118,
            "unit": "iter/sec",
            "range": "stddev: 0.000002388541362280538",
            "extra": "mean: 13.43823398243314 usec\nrounds: 21976"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 42572.268438730694,
            "unit": "iter/sec",
            "range": "stddev: 0.000003978144251942863",
            "extra": "mean: 23.489469475632557 usec\nrounds: 12105"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 73974.53180898007,
            "unit": "iter/sec",
            "range": "stddev: 0.000002533733127056375",
            "extra": "mean: 13.5181659896441 usec\nrounds: 23381"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.555949779717352,
            "unit": "iter/sec",
            "range": "stddev: 0.01916224420947513",
            "extra": "mean: 51.13533278946952 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 237.1551141819224,
            "unit": "iter/sec",
            "range": "stddev: 0.00006618106056148361",
            "extra": "mean: 4.216649526827818 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1919.3600185648522,
            "unit": "iter/sec",
            "range": "stddev: 0.000027419883210425815",
            "extra": "mean: 521.0069972947139 usec\nrounds: 1478"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1907.7338906889809,
            "unit": "iter/sec",
            "range": "stddev: 0.00002767805082379846",
            "extra": "mean: 524.1821225070591 usec\nrounds: 1053"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4214.2359116235,
            "unit": "iter/sec",
            "range": "stddev: 0.000013708525079738026",
            "extra": "mean: 237.29093979808977 usec\nrounds: 3754"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 120.79825793319664,
            "unit": "iter/sec",
            "range": "stddev: 0.00023021863913080492",
            "extra": "mean: 8.27826507691043 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3215.6839649105505,
            "unit": "iter/sec",
            "range": "stddev: 0.000011038269698567492",
            "extra": "mean: 310.9758331079704 usec\nrounds: 2223"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 41042.35835783948,
            "unit": "iter/sec",
            "range": "stddev: 0.000006743909662225894",
            "extra": "mean: 24.365071599473293 usec\nrounds: 9204"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 33874.38720947922,
            "unit": "iter/sec",
            "range": "stddev: 0.000008398799974078112",
            "extra": "mean: 29.52082922758129 usec\nrounds: 14247"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 37970.61966007137,
            "unit": "iter/sec",
            "range": "stddev: 0.000007454215215345671",
            "extra": "mean: 26.336151712887812 usec\nrounds: 14132"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "fd2fa79f18616e14f32a5fefe0cb011f224ee210",
          "message": "asdex `v0.1.8`\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-02T01:12:59+02:00",
          "tree_id": "928eeeeb774fbd3b9812a12b9e2bc7ddce09a9f9",
          "url": "https://github.com/adrhill/asdex/commit/fd2fa79f18616e14f32a5fefe0cb011f224ee210"
        },
        "date": 1775085220869,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 800.4428561402693,
            "unit": "iter/sec",
            "range": "stddev: 0.0038318006763784973",
            "extra": "mean: 1.2493084201188254 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3257.4476461629065,
            "unit": "iter/sec",
            "range": "stddev: 0.00000807257896170929",
            "extra": "mean: 306.9888172041521 usec\nrounds: 2139"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 74473.71305710521,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026410909663845102",
            "extra": "mean: 13.42755663643649 usec\nrounds: 25187"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 43550.96057259055,
            "unit": "iter/sec",
            "range": "stddev: 0.000003814768555488373",
            "extra": "mean: 22.9616060553522 usec\nrounds: 11692"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 79562.6012030468,
            "unit": "iter/sec",
            "range": "stddev: 0.000002394031797836296",
            "extra": "mean: 12.568719283674017 usec\nrounds: 26753"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.161512663392944,
            "unit": "iter/sec",
            "range": "stddev: 0.011360434919909613",
            "extra": "mean: 45.12327363157977 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.91191189685694,
            "unit": "iter/sec",
            "range": "stddev: 0.000023852479531751348",
            "extra": "mean: 4.150894790242402 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1962.0236970562426,
            "unit": "iter/sec",
            "range": "stddev: 0.000026452191024576955",
            "extra": "mean: 509.6778400283177 usec\nrounds: 1394"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1934.7915252721216,
            "unit": "iter/sec",
            "range": "stddev: 0.000026087354413335162",
            "extra": "mean: 516.8515506389524 usec\nrounds: 1017"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4070.529740528044,
            "unit": "iter/sec",
            "range": "stddev: 0.000011427544460128633",
            "extra": "mean: 245.6682701623687 usec\nrounds: 4005"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 101.07899623116694,
            "unit": "iter/sec",
            "range": "stddev: 0.009646667322144371",
            "extra": "mean: 9.89325218181834 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3218.5259767611697,
            "unit": "iter/sec",
            "range": "stddev: 0.000007658861307615038",
            "extra": "mean: 310.7012362865279 usec\nrounds: 2133"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 42026.61112520322,
            "unit": "iter/sec",
            "range": "stddev: 0.000007811108637921479",
            "extra": "mean: 23.794447689842478 usec\nrounds: 13831"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 37353.08618964929,
            "unit": "iter/sec",
            "range": "stddev: 0.000007505666249586156",
            "extra": "mean: 26.771549609657278 usec\nrounds: 13959"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 45790.29878897619,
            "unit": "iter/sec",
            "range": "stddev: 0.000006455553728064844",
            "extra": "mean: 21.83868693690956 usec\nrounds: 15425"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "me@sirmarcel.com",
            "name": "Marcel",
            "username": "sirmarcel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "14d3a056c9b1801efdaf6b2fdbef2ca7f93a3fff",
          "message": "fix(coloring): enforce star constraint for internal vertices (#102)\n\n* fix(coloring): forbid 2-colored P4s with the current vertex internal\n\n`color_symmetric` produced invalid star colorings on a large fraction of\ngraphs because the inner star-constraint check only covered one of the two\ncases of a 2-colored 4-vertex path involving the vertex being colored.\n\nA star coloring (Gebremedhin et al., 2005) is a distance-1 coloring with\nthe extra constraint that every path on 4 vertices uses >=3 distinct\ncolors. When assigning a color to vertex `v`, any 2-colored 4-path\ninvolving `v` must be forbidden. Such a path has the pattern `(a,b,a,b)`,\nand `v` can sit at any of the 4 positions. By symmetry of undirected paths\nthere are two distinct cases:\n\n- Case A (`v` is an endpoint): path `v-w-u-x`, violation when\n  `c(v) = c(u)` and `c(w) = c(x)`. Handled by the existing code: for each\n  colored neighbor `w` of `v`, and each colored neighbor `u` of `w`\n  (`u != v`), forbid `c(u)` if `u` has another colored neighbor `x != w`\n  with `c(x) = c(w)`.\n\n- Case B (`v` is internal): path `a-v-w-x`, violation when `c(a) = c(w)`\n  and `c(v) = c(x)`. Equivalently: `v` has two colored neighbors (`a` and\n  `w`) sharing color `c(w)`, and `w` has any colored neighbor `x != v`.\n  This case was not checked, so the algorithm happily assigned `c(x)` to\n  `v` and produced an invalid coloring.\n\nFix: maintain `neighbor_color_count[c]` for `v`, and when iterating\ncolored neighbors `w` of `v`, short-circuit-forbid every colored\nneighbor of `w` (other than `v`) whenever `neighbor_color_count[c(w)] >= 2`.\n\nEvidence: an Erdos-Renyi fuzz over ~3000 random graphs (n in [6,14],\np in [0.2,0.6]) showed the pre-fix implementation failed the star\nconstraint on ~46% of graphs. After the fix: 0 failures. The bug also\nsurfaced on a real MACE-MP-0 Hessian sparsity pattern for ZIF-8 2x2x2,\nwhere ~1.1% of colors contained seeds that violated the star constraint,\ncontaminating ~0.95% of extracted entries.\n\nTests added:\n- `test_star_case_b_internal_vertex`: minimal 12-vertex counterexample.\n- `test_star_random_graphs[seed=0..19]`: parametrized Erdos-Renyi fuzz.\n\nBoth use `_is_valid_star_coloring` (brute-force P4 enumeration) as the\nunambiguous spec.\n\n* fix(coloring): mirror SMC's Case 1 / Case 2 structure in `color_symmetric`\n\nReplace the `neighbor_color_count` approach with a `first_neighbor` dict,\nmirroring SparseMatrixColorings.jl's `star_coloring` algorithm structure.\n\nCase 1 (v internal): when a second neighbor with color `cw` is seen,\nforbid all colored neighbors of both that neighbor and the previously-recorded\nfirst neighbor.\nCase 2 (v endpoint): on first encounter of color `cw`, scan for 2-colored P4s\nvia the hub-equivalent neighbor check.\n\nThe distance-1 forbid is now folded into the single neighbor loop\ninstead of a separate pass.\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>\n\n* test(coloring): simplify star-coloring tests and make fuzz actually fuzz\n\nExtract a `_make_symmetric_reflexive_graph` helper to deduplicate the\ndiagonal-plus-both-edge-directions construction shared by the two new\nstar tests, and convert `test_star_random_graphs` into a real fuzz test:\nit now draws fresh OS entropy via `np.random.SeedSequence()` per run and\nreports the seed on failure so any counterexample can be replayed with\n`np.random.default_rng(<entropy>)`. Replace the silent `return` on empty\nrandom graphs with `pytest.skip`.\n\nAlso record two follow-up items in `PLAN.md`: building SMC-style\nstar/hub data in `color_symmetric` (to unlock postprocessing and O(1)\nCase 2 checks in `decompression.py`) and exposing a public\n`check_star_coloring` in `verify.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: adrhill <adrian.hill@mailbox.org>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-16T15:55:18+02:00",
          "tree_id": "731be3d228166d3aec9d444e7b958aeccf2f2ab6",
          "url": "https://github.com/adrhill/asdex/commit/14d3a056c9b1801efdaf6b2fdbef2ca7f93a3fff"
        },
        "date": 1776347753821,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 746.8289385740733,
            "unit": "iter/sec",
            "range": "stddev: 0.003971531988913668",
            "extra": "mean: 1.3389947126437123 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3077.7633555588563,
            "unit": "iter/sec",
            "range": "stddev: 0.00000926935263412074",
            "extra": "mean: 324.9112698004754 usec\nrounds: 1553"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 72415.1497739132,
            "unit": "iter/sec",
            "range": "stddev: 0.0000034829401548422717",
            "extra": "mean: 13.809265093313932 usec\nrounds: 19661"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 42687.00834763106,
            "unit": "iter/sec",
            "range": "stddev: 0.000004710028089167122",
            "extra": "mean: 23.42633130568157 usec\nrounds: 11672"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 76695.17137021874,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029332029338628992",
            "extra": "mean: 13.038630491779653 usec\nrounds: 22511"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.98307034280565,
            "unit": "iter/sec",
            "range": "stddev: 0.01920967345470824",
            "extra": "mean: 50.04235999999981 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 179.75940874901525,
            "unit": "iter/sec",
            "range": "stddev: 0.00007535248806700082",
            "extra": "mean: 5.562991149999975 msec\nrounds: 140"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1889.1925408229833,
            "unit": "iter/sec",
            "range": "stddev: 0.000027018769230041007",
            "extra": "mean: 529.3266717877115 usec\nrounds: 1432"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1876.076366845533,
            "unit": "iter/sec",
            "range": "stddev: 0.000033447623224858365",
            "extra": "mean: 533.0273424217891 usec\nrounds: 1247"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 3777.931806556929,
            "unit": "iter/sec",
            "range": "stddev: 0.000019548078840775596",
            "extra": "mean: 264.6950901189939 usec\nrounds: 3107"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 99.18529302911566,
            "unit": "iter/sec",
            "range": "stddev: 0.011379143073972588",
            "extra": "mean: 10.082139896551515 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 2996.915437767556,
            "unit": "iter/sec",
            "range": "stddev: 0.000010362931966178872",
            "extra": "mean: 333.67641522275113 usec\nrounds: 1616"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 41734.118788390595,
            "unit": "iter/sec",
            "range": "stddev: 0.00000877162805690451",
            "extra": "mean: 23.961210372511218 usec\nrounds: 9583"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 38274.13091192315,
            "unit": "iter/sec",
            "range": "stddev: 0.000008702722328349018",
            "extra": "mean: 26.1273078231668 usec\nrounds: 9996"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 42610.28238873886,
            "unit": "iter/sec",
            "range": "stddev: 0.000007430339498000066",
            "extra": "mean: 23.468513793850903 usec\nrounds: 14173"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a22653e6f6b297131bcacf804423f82fcb2df77b",
          "message": "refactor(coloring): tidy coloring tests and move validators (#103)\n\n* test(coloring): inline `_make_pattern` and `_from_dense` helpers\n\nReplace the thin helpers with direct calls to\n`SparsityPattern.from_coo(...)` and\n`SparsityPattern.from_dense(np.array(...))`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(coloring): move coloring validators to `coloring` module\n\nRelocate `_is_valid_row_coloring`, `_is_valid_col_coloring`, and\n`_is_valid_star_coloring` from `tests/test_coloring.py` to\n`src/asdex/coloring.py` so they live next to the algorithms they verify.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-16T16:22:17+02:00",
          "tree_id": "0b980780806d02b6646c8cbf9c76578f9a01bd7d",
          "url": "https://github.com/adrhill/asdex/commit/a22653e6f6b297131bcacf804423f82fcb2df77b"
        },
        "date": 1776349372994,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 778.7125686405179,
            "unit": "iter/sec",
            "range": "stddev: 0.004073997850617014",
            "extra": "mean: 1.2841708741722344 msec\nrounds: 151"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3303.0184316296672,
            "unit": "iter/sec",
            "range": "stddev: 0.000009758030042690929",
            "extra": "mean: 302.7533817020248 usec\nrounds: 2033"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 78666.16872972532,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024599426878467224",
            "extra": "mean: 12.71194486966458 usec\nrounds: 25739"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 44098.07827240699,
            "unit": "iter/sec",
            "range": "stddev: 0.0000044012519804544454",
            "extra": "mean: 22.676725135791667 usec\nrounds: 12890"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80100.30356784654,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024141252268418327",
            "extra": "mean: 12.484347192928928 usec\nrounds: 24652"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.331500209961874,
            "unit": "iter/sec",
            "range": "stddev: 0.01902328807443438",
            "extra": "mean: 49.18476205263139 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 233.2599711659731,
            "unit": "iter/sec",
            "range": "stddev: 0.00025683625275016154",
            "extra": "mean: 4.28706217788419 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1585.4156497237586,
            "unit": "iter/sec",
            "range": "stddev: 0.00009386521515781764",
            "extra": "mean: 630.7494190398834 usec\nrounds: 1229"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1914.7881998032617,
            "unit": "iter/sec",
            "range": "stddev: 0.00003246024532995905",
            "extra": "mean: 522.2509727722088 usec\nrounds: 1212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4264.218839569235,
            "unit": "iter/sec",
            "range": "stddev: 0.000011642513451811812",
            "extra": "mean: 234.50954034549937 usec\nrounds: 3879"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 109.12864418874383,
            "unit": "iter/sec",
            "range": "stddev: 0.009763834164542916",
            "extra": "mean: 9.163496966666665 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3287.2802235572267,
            "unit": "iter/sec",
            "range": "stddev: 0.000008827641155166721",
            "extra": "mean: 304.20284611997016 usec\nrounds: 2268"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 37595.68947901527,
            "unit": "iter/sec",
            "range": "stddev: 0.00000799134205100274",
            "extra": "mean: 26.59879400690785 usec\nrounds: 14350"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 34098.37497000866,
            "unit": "iter/sec",
            "range": "stddev: 0.000008087464058454887",
            "extra": "mean: 29.32691076567588 usec\nrounds: 13896"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 45129.37529115349,
            "unit": "iter/sec",
            "range": "stddev: 0.000007092027335592206",
            "extra": "mean: 22.158516344364855 usec\nrounds: 11686"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fd9241e4ed555394503e78488a61ea4a22ff35c7",
          "message": "feat(coloring)!: JIT with numba, port SMC graph layer, track star sets (#104)\n\n* feat(coloring)!: track star sets and add public coloring validators\n\n- `color_symmetric` now returns `(colors, num_colors, star_set)` instead of\n  `(colors, num_colors)`. The new `StarSet` stores per-edge star assignment\n  and per-star hub vertex, enabling O(1) hub lookup during Hessian\n  decompression.\n- Add `postprocess=True` (default) which demotes colors never needed as a\n  hub to neutral (`-1`), reducing HVP count.\n- Add `forced_colors` kwarg to verify a user-supplied coloring against the\n  star-coloring constraints; raises the new `InvalidColoringError` on\n  violation.\n- Promote private validators to public API: `check_coloring_rows`,\n  `check_coloring_cols`, `check_coloring_symmetric` (in `verify.py`).\n- `ColoredPattern` gains an optional `star_set` field; decompression uses\n  hub-based extraction when present and falls back to the existing\n  uniqueness heuristic otherwise.\n- `color_rows` and `color_cols` are unchanged.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(coloring): polish docstrings and simplify tests\n\n- Drop `SMC` reference from `color_symmetric` docstring and cite\n  Gebremedhin et al. (2007), Algorithm 4.1 directly.\n- Add backticks to `True` in the `postprocess` docstring.\n- Rename section headers from `Public API: ...` to plain descriptive\n  headers so they don't imply `StarSet` above isn't public.\n- Add a permalink to the SMC stamp-trick line in the source comment.\n- Drop the three `_is_valid_*_coloring` test helpers; call\n  `check_coloring_rows` / `check_coloring_cols` /\n  `check_coloring_symmetric` directly — the validators already raise\n  `InvalidColoringError` with the specific violation.\n- Drop seed reporting from the fuzz test since the validator error\n  already identifies the offending vertex/edge.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(coloring): fold hub encoding into StarSet docstring\n\n- Inline the trivial-star hub encoding notes (previously a floating\n  comment block) into the `StarSet.hub` attribute docstring.\n- Drop the `(Gebremedhin et al., 2007)` citation from `_update_stars`:\n  `_update_stars!` is SMC's own function name, not from the paper.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(coloring): split coloring.py into private submodules\n\nGroups the module's contents into five private submodules behind an\n`__init__.py` re-export:\n\n- `_types.py`: `StarSet`, `DenseColoringWarning`, `InvalidColoringError`\n- `_color_greedy.py`: `color_rows`, `color_cols` + conflict/greedy\n  helpers\n- `_color_symmetric.py`: `color_symmetric` + adjacency/star helpers\n- `_postprocessing.py`: `_postprocess_star_coloring`\n- `_api.py`: `jacobian_coloring*`, `hessian_coloring*` + empty/dense\n  helpers\n\nThe public surface of `asdex.coloring` is unchanged — every name in\n`__all__` still resolves via `__init__.py` re-exports.\n`tests/test_coloring.py` moves its `_greedy_color` import to\n`asdex.coloring._color_greedy`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(coloring): add SMC attribution and source file links\n\nAdd the SparseMatrixColorings.jl attribution block plus a link to the\ncorresponding SMC source file in each submodule docstring of\n`src/asdex/coloring/`. Also correct `_types.py`, which previously claimed\n`StarSet` lived in `forest.jl` — it is actually defined in `coloring.jl`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(detection): nest `_interpret/` under `detection/`\n\nConvert `detection.py` into a package mirroring `coloring/`:\n`detection.py` → `detection/_api.py`,\n`_interpret/` → `detection/_interpret/`.\n\nThe interpreter is consumed only by `detection`,\nso nesting it under its sole consumer reflects the dependency\ndirection and keeps the package root focused on pipeline stages.\n\nUpdates imports in `decompression.py` and `tests/_interpret/test_internals.py`,\nand path references in `CLAUDE.md`, `tests/CLAUDE.md`,\nand `.claude/skills/add-handler/SKILL.md`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf(coloring): port SMC's CSR graph layer\n\nIntroduces `_graph.py` mirroring SparseMatrixColorings.jl's `graph.jl`\n(`_build_csr`, `_build_symmetric_csr`, `_build_edge_to_index`) and\nrewrites both coloring paths on top of it.\n\n- `_color_greedy.py` now mirrors `partial_distance2_coloring` faithfully.\n  Drops the O(Σ deg²) `list[set[int]]` conflict graphs and per-vertex\n  `set[int]` allocations for SMC's `forbidden_colors` timestamp trick.\n  LargestFirst uses SMC's `visited`-timestamp trick. `color_rows` and\n  `color_cols` share one `_partial_distance2_coloring` helper.\n- `_color_symmetric.py`'s `_build_adjacency_with_edge_index`\n  (list-of-tuples + `dict[(i, j), int]`) is replaced by\n  `_build_symmetric_csr` + `_build_edge_to_index`; `_treat` and\n  `_update_stars` iterate CSR slices.\n\nPublic API and `StarSet` contract unchanged. `_greedy_color` removed;\nits zero-vertex test is covered by `test_color_zero_row_pattern`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf(coloring): run hot loops on Python lists, not numpy scalars\n\nThe SMC port in `_color_greedy.py` and `_color_symmetric.py` iterated CSR\narrays (`indptr`, `neighbors`, `edge_to_index`) and stamp buffers\n(`forbidden`, `visited`, `colors`, `star`) as numpy arrays, with per-element\n`int(...)` boxing in the triple-nested loops. That boxing dominates in\nCPython and made `color_rows` 4-7x slower than `main`'s conflict-set approach.\n\nRebind the hot arrays to Python lists via `.tolist()` at the top of each\nfunction, run the loop in unboxed-int territory, and re-wrap as numpy only\nat the public boundary. No algorithmic change; all 93 coloring tests pass.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf(coloring): JIT hot loops with numba\n\nCloses the 1.2-1.6× perf gap vs `main` left by the list-based port in\n0a12bf0 and goes further — post-compile timings beat `main` by 6-9×:\n\n| bench      | main    | previous | new      |\n| ---------- | ------- | -------- | -------- |\n| convnet    | 2.22 ms | 3.53 ms  | 245 µs   |\n| heat       | 158 µs  | 197 µs   | 23.5 µs  |\n| rosenbrock | 159 µs  | 194 µs   | 21.5 µs  |\n\nChanges:\n\n- Add `numba>=0.63.0` as a required dependency; ignore `*.nbi` / `*.nbc`.\n- `@njit(cache=True)` the hot kernels: `_build_edge_to_index_core` in\n  `_graph.py`; `_distance2_degrees` and `_partial_distance2_coloring`\n  in `_color_greedy.py`; `_star_coloring_core` (plus `_treat` and\n  `_update_stars`) in `_color_symmetric.py`.\n- Drop the `.tolist()` rebinding; numba reads int32 arrays directly.\n- Refactor for numba compatibility in `_color_symmetric.py`:\n  preallocated `hub_buf` + `hub_len` counter instead of `hub_list`;\n  three parallel `int32[n]` arrays (`fn_p`, `fn_q`, `fn_edge`) instead\n  of a `first_neighbor` tuple list; `InvalidColoringError` piped via\n  return sentinels so the jitted core never raises.\n\nCo-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>\n\n* test(benchmarks): warm up numba JIT before coloring timings\n\nCall `color_rows(sparsity)` once before `benchmark(...)` in the three\ncoloring benchmarks so the first-call compile pass is never folded into\nthe measured rounds. Previously the timings relied on pytest-benchmark's\ncalibration phase incidentally triggering the JIT — now the warmup is\nexplicit, matching the `# warmup` pattern used by the materialization\nand end-to-end benchmarks.\n\nCo-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>\n\n* feat(coloring)!: default `postprocess` to `False`, expose via API\n\nAlign `color_symmetric`'s default with SparseMatrixColorings.jl's\n`GreedyColoringAlgorithm` (`postprocessing=false`), and thread a\n`postprocess` kwarg through `jacobian_coloring`, `hessian_coloring`,\nand their `_from_sparsity` counterparts so callers can opt in.\n\nAdd tests exercising both paths on a 4-cycle without a diagonal\n(where postprocessing strictly reduces 3 colors to 2) and verifying\nthe no-op case on a banded pattern with full diagonal.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(coloring): reach 100% coverage and drop dead defensive branches\n\nAdd tests covering the previously unreached branches of\n`jacobian_coloring_from_sparsity`, `hessian_coloring_from_sparsity`,\n`color_symmetric(forced_colors=...)`, `StarSet.hub_vertex`, and both\nbranches of trivial-star postprocessing. Each new test makes concrete\nassertions (mode resolution, HVP roundtrips, forced-coloring violations,\nhub-flip identity) instead of merely exercising code paths.\n\nRemove guards that are unreachable under their public-API contracts:\nthe non-square check in `_empty_hessian_pattern`, the unused\n`nb_self_loops` counter in `_build_edge_to_index`, and the\n`num_colors == 0` / `ci < 0` / `s < 0` branches in\n`_postprocess_star_coloring`. Coverage (with `NUMBA_DISABLE_JIT=1`) is\nnow 100% across `src/asdex/coloring/`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* chore: bump version to `0.2.0-DEV`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-16T23:28:14+02:00",
          "tree_id": "93c0e6b2726457fb3683b4da117e83ecc57649a8",
          "url": "https://github.com/adrhill/asdex/commit/fd9241e4ed555394503e78488a61ea4a22ff35c7"
        },
        "date": 1776374932831,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 739.8572653987204,
            "unit": "iter/sec",
            "range": "stddev: 0.004865715617966829",
            "extra": "mean: 1.3516120564972554 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 26721.14456113902,
            "unit": "iter/sec",
            "range": "stddev: 0.000004085139815989185",
            "extra": "mean: 37.423546649057684 usec\nrounds: 8371"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 76688.71673180048,
            "unit": "iter/sec",
            "range": "stddev: 0.000002705751765836641",
            "extra": "mean: 13.039727910655342 usec\nrounds: 21945"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 44875.97813569258,
            "unit": "iter/sec",
            "range": "stddev: 0.000004077232408786841",
            "extra": "mean: 22.28363684856686 usec\nrounds: 12185"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 75483.69599841486,
            "unit": "iter/sec",
            "range": "stddev: 0.000004259256236939455",
            "extra": "mean: 13.24789395608026 usec\nrounds: 9977"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.517179115902355,
            "unit": "iter/sec",
            "range": "stddev: 0.0028261574764994516",
            "extra": "mean: 44.4105362777777 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2882.4577086942645,
            "unit": "iter/sec",
            "range": "stddev: 0.000011566816506686418",
            "extra": "mean: 346.9261654676605 usec\nrounds: 2502"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1909.6549420669846,
            "unit": "iter/sec",
            "range": "stddev: 0.000028902293059822085",
            "extra": "mean: 523.6548121712573 usec\nrounds: 1331"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1917.2734940400505,
            "unit": "iter/sec",
            "range": "stddev: 0.000026212286486559982",
            "extra": "mean: 521.5739971936996 usec\nrounds: 1069"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4155.50667052297,
            "unit": "iter/sec",
            "range": "stddev: 0.00002369661782994947",
            "extra": "mean: 240.64454211889165 usec\nrounds: 3870"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 114.14813322489044,
            "unit": "iter/sec",
            "range": "stddev: 0.010835619903795094",
            "extra": "mean: 8.76054624590169 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 26848.96231638439,
            "unit": "iter/sec",
            "range": "stddev: 0.000003943082337539176",
            "extra": "mean: 37.245387297137995 usec\nrounds: 19476"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 45580.296425913504,
            "unit": "iter/sec",
            "range": "stddev: 0.000007632732955811793",
            "extra": "mean: 21.939304445406716 usec\nrounds: 15589"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 40733.296451553004,
            "unit": "iter/sec",
            "range": "stddev: 0.000007456497290145844",
            "extra": "mean: 24.54994039555259 usec\nrounds: 14311"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 45168.40239923045,
            "unit": "iter/sec",
            "range": "stddev: 0.000007506121768396159",
            "extra": "mean: 22.139370597199544 usec\nrounds: 9795"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "68ca9673dc5bbf8dd67db6dbb6c726f4aec3224f",
          "message": "asdex \\`v0.2.0\\`",
          "timestamp": "2026-04-21T00:18:58+02:00",
          "tree_id": "d01b1686c3535d42b9da61ca21399da4b0343652",
          "url": "https://github.com/adrhill/asdex/commit/68ca9673dc5bbf8dd67db6dbb6c726f4aec3224f"
        },
        "date": 1776723585802,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 750.0506654860975,
            "unit": "iter/sec",
            "range": "stddev: 0.004893328328760484",
            "extra": "mean: 1.3332432674423584 msec\nrounds: 172"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 26258.319006993093,
            "unit": "iter/sec",
            "range": "stddev: 0.000006062575007669644",
            "extra": "mean: 38.08316898479605 usec\nrounds: 8196"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 79779.23569669962,
            "unit": "iter/sec",
            "range": "stddev: 0.000002478251914078504",
            "extra": "mean: 12.534589875011411 usec\nrounds: 24573"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 43594.18955115638,
            "unit": "iter/sec",
            "range": "stddev: 0.000004116302699460274",
            "extra": "mean: 22.93883681050045 usec\nrounds: 11839"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78629.93750107019,
            "unit": "iter/sec",
            "range": "stddev: 0.000002649221885617917",
            "extra": "mean: 12.717802300000423 usec\nrounds: 24087"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 18.727672763804435,
            "unit": "iter/sec",
            "range": "stddev: 0.031224727188526293",
            "extra": "mean: 53.39691763157735 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2866.4306752431276,
            "unit": "iter/sec",
            "range": "stddev: 0.000015026311931459164",
            "extra": "mean: 348.8659288490139 usec\nrounds: 2572"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1875.6082792901793,
            "unit": "iter/sec",
            "range": "stddev: 0.00003643867404998816",
            "extra": "mean: 533.1603677812982 usec\nrounds: 1316"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1926.419743136224,
            "unit": "iter/sec",
            "range": "stddev: 0.00003059854943142247",
            "extra": "mean: 519.0976699460074 usec\nrounds: 1118"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4092.9401754516466,
            "unit": "iter/sec",
            "range": "stddev: 0.000018916171823707064",
            "extra": "mean: 244.3231410998213 usec\nrounds: 3983"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 115.79637546788311,
            "unit": "iter/sec",
            "range": "stddev: 0.011028905661020648",
            "extra": "mean: 8.635848885247334 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 26594.63036916698,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038504087184799755",
            "extra": "mean: 37.601575435294265 usec\nrounds: 18897"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 44343.580816926595,
            "unit": "iter/sec",
            "range": "stddev: 0.00000655388830172621",
            "extra": "mean: 22.551178357212986 usec\nrounds: 15839"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 39865.92513707461,
            "unit": "iter/sec",
            "range": "stddev: 0.0000059286550771908485",
            "extra": "mean: 25.08407861003124 usec\nrounds: 13726"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 44145.65094006382,
            "unit": "iter/sec",
            "range": "stddev: 0.000006455287273788513",
            "extra": "mean: 22.65228802170551 usec\nrounds: 15478"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "9d65176f3dc0a75719be8ffd54a6b74fee51d7cb",
          "message": "asdex `v0.2.0`\n\nCo-Authored-By: Claude Code <noreply@anthropic.com>",
          "timestamp": "2026-04-21T00:24:04+02:00",
          "tree_id": "d01b1686c3535d42b9da61ca21399da4b0343652",
          "url": "https://github.com/adrhill/asdex/commit/9d65176f3dc0a75719be8ffd54a6b74fee51d7cb"
        },
        "date": 1776723941098,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 858.6558910346579,
            "unit": "iter/sec",
            "range": "stddev: 0.003505409929445766",
            "extra": "mean: 1.1646108882977861 msec\nrounds: 188"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 28851.585416621558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017791075453723525",
            "extra": "mean: 34.66014035484839 usec\nrounds: 8792"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 85018.18172417047,
            "unit": "iter/sec",
            "range": "stddev: 9.351756525449317e-7",
            "extra": "mean: 11.762189918908867 usec\nrounds: 20593"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 52917.40197943727,
            "unit": "iter/sec",
            "range": "stddev: 0.000002040748273409343",
            "extra": "mean: 18.89737520350265 usec\nrounds: 12284"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82694.04792905277,
            "unit": "iter/sec",
            "range": "stddev: 9.93773990269467e-7",
            "extra": "mean: 12.092768766839766 usec\nrounds: 30506"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.41979711258808,
            "unit": "iter/sec",
            "range": "stddev: 0.016913503214763027",
            "extra": "mean: 42.6989181499998 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 3511.9223248145536,
            "unit": "iter/sec",
            "range": "stddev: 0.000007430549140257112",
            "extra": "mean: 284.7443387156363 usec\nrounds: 2725"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2031.6191546182058,
            "unit": "iter/sec",
            "range": "stddev: 0.00003999837482374889",
            "extra": "mean: 492.2182377178493 usec\nrounds: 1262"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1987.495439516589,
            "unit": "iter/sec",
            "range": "stddev: 0.00003267386558606276",
            "extra": "mean: 503.1458085977929 usec\nrounds: 977"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4963.258630662386,
            "unit": "iter/sec",
            "range": "stddev: 0.00003064904895026438",
            "extra": "mean: 201.48053414386388 usec\nrounds: 3866"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 127.42921599740046,
            "unit": "iter/sec",
            "range": "stddev: 0.00835690382120612",
            "extra": "mean: 7.847493937500172 msec\nrounds: 64"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 28908.499020216663,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016968466463005118",
            "extra": "mean: 34.591903208141915 usec\nrounds: 20074"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 48457.84152586732,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033741284027361927",
            "extra": "mean: 20.636494910038227 usec\nrounds: 14342"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 41999.510952670054,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024019732625249718",
            "extra": "mean: 23.80980105046739 usec\nrounds: 14089"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 48505.968697032455,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027916263536611794",
            "extra": "mean: 20.616019571652817 usec\nrounds: 17883"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "479e5ea432e800c9f76cd51887435f8f2039429b",
          "message": "feat(api)!: require sample inputs, support multi-input functions (#105)\n\n## Summary\n\n- Support pytree / multi-positional-input functions across detection, coloring, and decompression via sample inputs (`*args`) and `argnums=`, mirroring `jax.make_jaxpr(f)(*args)` / `jax.jacrev` semantics.\n- Add `has_aux`, `holomorphic`, and `allow_int` kwargs to `jacobian`, `value_and_jacobian`, `hessian`, `value_and_hessian`, their `*_from_coloring` counterparts, and `jacobian_coloring` / `hessian_coloring`, matching `jax.jacrev` / `jax.jacfwd` / `jax.hessian` semantics.\n\n## Details\n\n- Introduce `_api_utils.py` helpers mirroring `jax._src.api_util`: `_ensure_index`, `_ensure_inbounds`, `avals_from_args`.\n- Generalize `SparsityPattern` to represent multi-input Jacobian / Hessian blocks.\n- Multi-input block structure for both Jacobian and Hessian decompression; `H[i][j]` / `J[i]` pytree returns.\n- Support PyTree outputs in Jacobian/Hessian computation.\n- Support scalar inputs (0-dimensional arrays).\n- PyTree input/output support in `check_jacobian_correctness` and `check_hessian_correctness`.\n- Dtype validators mirror `jax._src.api._check_{input,output}_dtype_{jacrev,jacfwd}` without coupling to JAX's private API.\n- `_jacobian_rows` seed dtype now matches the output dtype (needed for `allow_int` with int→float functions).\n- Replace the deprecated `jax.core.get_aval` with `jax.typeof`.\n- All multi-input tests verify structural and numerical equality against `jax.jacobian` / `jax.hessian`.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-05-13T15:12:50+02:00",
          "tree_id": "cfd18b564cde0670b840f1d3f0b2dd34b53937c7",
          "url": "https://github.com/adrhill/asdex/commit/479e5ea432e800c9f76cd51887435f8f2039429b"
        },
        "date": 1778678014637,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 1034.9466747443348,
            "unit": "iter/sec",
            "range": "stddev: 0.00005662084667546892",
            "extra": "mean: 966.2333571408712 usec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 27088.075161316247,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038280691068619045",
            "extra": "mean: 36.91661345609647 usec\nrounds: 8576"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 79639.73081695955,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025164970509096804",
            "extra": "mean: 12.556546710314178 usec\nrounds: 25626"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 45283.042930407995,
            "unit": "iter/sec",
            "range": "stddev: 0.00000423553059401689",
            "extra": "mean: 22.083321598701364 usec\nrounds: 10709"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78929.15401801828,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025289942363153784",
            "extra": "mean: 12.669589740841714 usec\nrounds: 27858"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.442559701064702,
            "unit": "iter/sec",
            "range": "stddev: 0.0006182463083892949",
            "extra": "mean: 42.65745774999914 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2889.451550880033,
            "unit": "iter/sec",
            "range": "stddev: 0.000009866669824098555",
            "extra": "mean: 346.086439724325 usec\nrounds: 2613"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1869.2289814770286,
            "unit": "iter/sec",
            "range": "stddev: 0.000028786145930887754",
            "extra": "mean: 534.9799355292572 usec\nrounds: 1427"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1888.076613187166,
            "unit": "iter/sec",
            "range": "stddev: 0.00003063616270052122",
            "extra": "mean: 529.6395246970147 usec\nrounds: 1073"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 3958.120816548925,
            "unit": "iter/sec",
            "range": "stddev: 0.00001675790908155969",
            "extra": "mean: 252.64514307370166 usec\nrounds: 3299"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 140.41793948704156,
            "unit": "iter/sec",
            "range": "stddev: 0.00010878283589816113",
            "extra": "mean: 7.121597166666049 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 26999.54378548048,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038834808966031395",
            "extra": "mean: 37.03766285628015 usec\nrounds: 19333"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 41160.62741986243,
            "unit": "iter/sec",
            "range": "stddev: 0.000010416345581806574",
            "extra": "mean: 24.2950621184516 usec\nrounds: 8516"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 41253.85303302762,
            "unit": "iter/sec",
            "range": "stddev: 0.000008039803457796861",
            "extra": "mean: 24.240160045157605 usec\nrounds: 14171"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 41619.03804022328,
            "unit": "iter/sec",
            "range": "stddev: 0.000008221008649114643",
            "extra": "mean: 24.027465484270362 usec\nrounds: 14660"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "9614d41c3fef7d30e543067985f3a69fabb204d2",
          "message": "asdex `v0.3.0`\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-05-13T15:18:17+02:00",
          "tree_id": "c1c155595f0ad8d401ccbdb13d868313e868b0ea",
          "url": "https://github.com/adrhill/asdex/commit/9614d41c3fef7d30e543067985f3a69fabb204d2"
        },
        "date": 1778678350250,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 1117.2475935176192,
            "unit": "iter/sec",
            "range": "stddev: 0.0000656196273638264",
            "extra": "mean: 895.0567500007148 usec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 27473.28095623775,
            "unit": "iter/sec",
            "range": "stddev: 0.000002519887687163709",
            "extra": "mean: 36.39900169160364 usec\nrounds: 7685"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 94322.68536940566,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012270998861665763",
            "extra": "mean: 10.601903413623106 usec\nrounds: 25076"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 66163.76809978182,
            "unit": "iter/sec",
            "range": "stddev: 0.000001716476400115386",
            "extra": "mean: 15.114012226327501 usec\nrounds: 10551"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 92815.61419305232,
            "unit": "iter/sec",
            "range": "stddev: 0.000001156249676314407",
            "extra": "mean: 10.774049266323281 usec\nrounds: 32781"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.88878628121611,
            "unit": "iter/sec",
            "range": "stddev: 0.023136663129276078",
            "extra": "mean: 47.87257557894751 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2844.185694969009,
            "unit": "iter/sec",
            "range": "stddev: 0.00002420300391324784",
            "extra": "mean: 351.59448335911003 usec\nrounds: 2584"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1898.3784655102634,
            "unit": "iter/sec",
            "range": "stddev: 0.000020820797244833316",
            "extra": "mean: 526.7653516767064 usec\nrounds: 1163"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1904.8495444597431,
            "unit": "iter/sec",
            "range": "stddev: 0.000017960157833856144",
            "extra": "mean: 524.9758454196558 usec\nrounds: 1048"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4533.607244318711,
            "unit": "iter/sec",
            "range": "stddev: 0.000017826399031282973",
            "extra": "mean: 220.5749078183051 usec\nrounds: 3786"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 103.11127581123573,
            "unit": "iter/sec",
            "range": "stddev: 0.016961424423661664",
            "extra": "mean: 9.69826037096743 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 27513.51581116711,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022249272443037933",
            "extra": "mean: 36.34577299619857 usec\nrounds: 20960"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 63797.119102394514,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024770140226494155",
            "extra": "mean: 15.674688983917875 usec\nrounds: 17774"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 53448.17799482231,
            "unit": "iter/sec",
            "range": "stddev: 0.000004070674651485548",
            "extra": "mean: 18.709711678045846 usec\nrounds: 14463"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 64406.19270785695,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031041335755760828",
            "extra": "mean: 15.526457285496544 usec\nrounds: 11015"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4992b90fda60cb540d2848781c6a89f8b0154fa4",
          "message": "Add find-bugs skill for testing against JAX references (#107)\n\n* claude: add find-bugs skill for testing against JAX references\n\n---------\n\nCo-authored-by: bdrhill <mail@adrianhill.de>\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-14T15:46:17+02:00",
          "tree_id": "29ede5eff6e6260e90cd8f83484d5c4ff00f10e0",
          "url": "https://github.com/adrhill/asdex/commit/4992b90fda60cb540d2848781c6a89f8b0154fa4"
        },
        "date": 1778766416195,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 989.3141480750074,
            "unit": "iter/sec",
            "range": "stddev: 0.00018401067387541732",
            "extra": "mean: 1.0108012727259434 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 27404.1107110314,
            "unit": "iter/sec",
            "range": "stddev: 0.000002491012597285307",
            "extra": "mean: 36.490875786655415 usec\nrounds: 7310"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 96779.61242907493,
            "unit": "iter/sec",
            "range": "stddev: 0.000001202365348150635",
            "extra": "mean: 10.332754749693292 usec\nrounds: 26528"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 65961.3600166294,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017666124630703494",
            "extra": "mean: 15.160390867439538 usec\nrounds: 8760"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 94825.02090693943,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011291281202017697",
            "extra": "mean: 10.545739831488067 usec\nrounds: 33240"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.782944787448443,
            "unit": "iter/sec",
            "range": "stddev: 0.02336466960197873",
            "extra": "mean: 48.11637668420961 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2850.322287235199,
            "unit": "iter/sec",
            "range": "stddev: 0.00000807303828209494",
            "extra": "mean: 350.83751913893076 usec\nrounds: 2508"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1909.0841472456907,
            "unit": "iter/sec",
            "range": "stddev: 0.000017995215431608482",
            "extra": "mean: 523.81137910696 usec\nrounds: 1187"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1904.6822168654687,
            "unit": "iter/sec",
            "range": "stddev: 0.000015292504516603047",
            "extra": "mean: 525.0219648953818 usec\nrounds: 1054"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4725.00114320947,
            "unit": "iter/sec",
            "range": "stddev: 0.000007766331678178898",
            "extra": "mean: 211.64016043406656 usec\nrounds: 3503"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.6823401024055,
            "unit": "iter/sec",
            "range": "stddev: 0.012092492447544395",
            "extra": "mean: 8.151132421873264 msec\nrounds: 64"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 27044.269812262915,
            "unit": "iter/sec",
            "range": "stddev: 0.0000025674699721468695",
            "extra": "mean: 36.976409677238216 usec\nrounds: 20626"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 63683.667261271636,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027757417241086147",
            "extra": "mean: 15.702613291683605 usec\nrounds: 16612"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 55355.732894682085,
            "unit": "iter/sec",
            "range": "stddev: 0.0000031905384429621804",
            "extra": "mean: 18.064976249209195 usec\nrounds: 14568"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 63974.22227554929,
            "unit": "iter/sec",
            "range": "stddev: 0.000002700896443740845",
            "extra": "mean: 15.631295925612154 usec\nrounds: 17278"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "f130126ef4852cba05eaceb4bdd47ab08c991943",
          "message": "claude: Add `find-bugs` skill for testing against JAX references (#107)\n\n* claude: add find-bugs skill for testing against JAX references\n\n---------\n\nCo-authored-by: bdrhill <mail@adrianhill.de>\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-14T16:23:33+02:00",
          "tree_id": "29ede5eff6e6260e90cd8f83484d5c4ff00f10e0",
          "url": "https://github.com/adrhill/asdex/commit/f130126ef4852cba05eaceb4bdd47ab08c991943"
        },
        "date": 1778768660426,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 642.7871047201362,
            "unit": "iter/sec",
            "range": "stddev: 0.006569475228886993",
            "extra": "mean: 1.5557250490197545 msec\nrounds: 102"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 28822.4012687453,
            "unit": "iter/sec",
            "range": "stddev: 0.000001730960542292387",
            "extra": "mean: 34.69523551059534 usec\nrounds: 8696"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 80886.36066044083,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012168413871606566",
            "extra": "mean: 12.363023776010621 usec\nrounds: 26287"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 53618.93553779154,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015137892465224078",
            "extra": "mean: 18.65012779478218 usec\nrounds: 12121"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84259.47910953891,
            "unit": "iter/sec",
            "range": "stddev: 8.934672571012572e-7",
            "extra": "mean: 11.86810090173927 usec\nrounds: 28057"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.12194147056143,
            "unit": "iter/sec",
            "range": "stddev: 0.02050042392114951",
            "extra": "mean: 45.203989050000004 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 3532.969419292185,
            "unit": "iter/sec",
            "range": "stddev: 0.00000726564140616512",
            "extra": "mean: 283.04802032516477 usec\nrounds: 2952"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1744.8089144248584,
            "unit": "iter/sec",
            "range": "stddev: 0.00022231164852950117",
            "extra": "mean: 573.1286628195788 usec\nrounds: 1213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1768.636289118928,
            "unit": "iter/sec",
            "range": "stddev: 0.00004725964052251508",
            "extra": "mean: 565.4073741176965 usec\nrounds: 850"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4567.677345645737,
            "unit": "iter/sec",
            "range": "stddev: 0.00003628050261390096",
            "extra": "mean: 218.9296494318447 usec\nrounds: 3520"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 108.43446601932122,
            "unit": "iter/sec",
            "range": "stddev: 0.014272511423628662",
            "extra": "mean: 9.222160044775954 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 28748.238028933105,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017894758968085637",
            "extra": "mean: 34.78474051152524 usec\nrounds: 19392"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 47670.606601905776,
            "unit": "iter/sec",
            "range": "stddev: 0.000004406293967846312",
            "extra": "mean: 20.97728708071489 usec\nrounds: 14846"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 43032.53173227368,
            "unit": "iter/sec",
            "range": "stddev: 0.000002797273636372155",
            "extra": "mean: 23.238233023831516 usec\nrounds: 13784"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 48369.48700261371,
            "unit": "iter/sec",
            "range": "stddev: 0.000003403981552544403",
            "extra": "mean: 20.67419073404611 usec\nrounds: 16987"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mail@adrianhill.de",
            "name": "Bdrian Hill",
            "username": "bdrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "27bd7110c10729c2cdb4f758fb940bb7ffa4daa5",
          "message": "docs(skill): improve find-bugs workflow and test coverage (#111)\n\n* Add find-bugs skill for testing against JAX references (#107)\n\n* claude: add find-bugs skill for testing against JAX references\n\n---------\n\nCo-authored-by: bdrhill <mail@adrianhill.de>\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(skill): improve find-bugs workflow and test coverage\n\n- Emphasize filing issues immediately upon finding bugs/missing features\n- Add support for feature requests (missing primitives)\n- Add reduction edge cases: empty axis, keepdims, partial axes\n- Add sparsity precision testing section\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(skill): check open issues first, skip re-confirming known bugs\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Adrian Hill <adrian.hill@mailbox.org>\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-14T19:25:46+02:00",
          "tree_id": "3cdb4b3232eb7fe232f7139bf984158e15192cda",
          "url": "https://github.com/adrhill/asdex/commit/27bd7110c10729c2cdb4f758fb940bb7ffa4daa5"
        },
        "date": 1778779590680,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 661.6353425899165,
            "unit": "iter/sec",
            "range": "stddev: 0.005550333944267139",
            "extra": "mean: 1.5114065643555004 msec\nrounds: 101"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 26406.94477950086,
            "unit": "iter/sec",
            "range": "stddev: 0.00000418980689403539",
            "extra": "mean: 37.86882611184458 usec\nrounds: 8724"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 77216.03672203292,
            "unit": "iter/sec",
            "range": "stddev: 0.000003039418292542034",
            "extra": "mean: 12.950677637080261 usec\nrounds: 27407"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 42932.07387386561,
            "unit": "iter/sec",
            "range": "stddev: 0.000005728619229518164",
            "extra": "mean: 23.29260876001469 usec\nrounds: 12146"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 68187.30404390628,
            "unit": "iter/sec",
            "range": "stddev: 0.000005327947411200984",
            "extra": "mean: 14.665486691717467 usec\nrounds: 24233"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.757731558256083,
            "unit": "iter/sec",
            "range": "stddev: 0.018820349969121155",
            "extra": "mean: 48.17482089473619 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2851.9720722713905,
            "unit": "iter/sec",
            "range": "stddev: 0.000010554260528346215",
            "extra": "mean: 350.63456957471954 usec\nrounds: 2235"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1914.1305074614843,
            "unit": "iter/sec",
            "range": "stddev: 0.00002607132720743308",
            "extra": "mean: 522.430417415058 usec\nrounds: 1447"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1861.594170075524,
            "unit": "iter/sec",
            "range": "stddev: 0.00003321221069259269",
            "extra": "mean: 537.1740071357391 usec\nrounds: 981"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4057.1348653952004,
            "unit": "iter/sec",
            "range": "stddev: 0.000018852812724006422",
            "extra": "mean: 246.47935875372764 usec\nrounds: 3370"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 118.00508481468009,
            "unit": "iter/sec",
            "range": "stddev: 0.009723906446096997",
            "extra": "mean: 8.474211103448976 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 26103.332225359038,
            "unit": "iter/sec",
            "range": "stddev: 0.000004309641440101107",
            "extra": "mean: 38.30928524246086 usec\nrounds: 18770"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 45273.749074252366,
            "unit": "iter/sec",
            "range": "stddev: 0.000008299748482104519",
            "extra": "mean: 22.087854892686806 usec\nrounds: 12825"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 41210.629826867735,
            "unit": "iter/sec",
            "range": "stddev: 0.00000765080683943648",
            "extra": "mean: 24.26558400590225 usec\nrounds: 13505"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 43895.977372628586,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075031773319375414",
            "extra": "mean: 22.781130751711018 usec\nrounds: 9193"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3bf2cacdb423345cfd38e8bbca09e7ec7434259e",
          "message": "feat(detection): add handlers for scalar special functions (#127)\n\nAdd handlers for 7 special function primitives, closing #116 and #119.\n\nUnary elementwise (diagonal Jacobian):\n- `erfc`: ∂/∂x = -2/√π · e^(-x²)\n- `erf_inv`: ∂/∂x = √π/2 · e^(erf_inv(x)²)\n- `digamma`: ∂/∂x = ψ₁(x) (trigamma)\n- `lgamma`: ∂/∂x = ψ(x) (digamma)\n- `bessel_i0e`: ∂/∂x = I₁e(x) - sign(x)·I₀e(x)\n- `bessel_i1e`: ∂/∂x = I₀e(x) - I₁e(x)/x - sign(x)·I₁e(x)\n\nBinary with zero first derivative:\n- `polygamma(n, x)`: ∂/∂n = 0, ∂/∂x = ψₙ₊₁(x)\n\nAlso adds `is_der1_zero_globally` and `is_der2_zero_globally` kwargs\nto `prop_binary_const` for precise handling of binary ops where one\npartial derivative is zero everywhere (naming follows SparseConnectivityTracer).\n\nCloses #116, closes #119.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-15T13:13:41+02:00",
          "tree_id": "b114e6fd33cd128b27164e6e6bd0d8cf8ebfe6c7",
          "url": "https://github.com/adrhill/asdex/commit/3bf2cacdb423345cfd38e8bbca09e7ec7434259e"
        },
        "date": 1778843660483,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 998.7011891025976,
            "unit": "iter/sec",
            "range": "stddev: 0.000054322538299099766",
            "extra": "mean: 1.0013005000009758 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 26999.208450354992,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043089816624077866",
            "extra": "mean: 37.03812287085223 usec\nrounds: 6869"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 80960.38387611028,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027834882821697583",
            "extra": "mean: 12.351720089794178 usec\nrounds: 22275"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 45967.46148382998,
            "unit": "iter/sec",
            "range": "stddev: 0.000003906830342970013",
            "extra": "mean: 21.754518690395184 usec\nrounds: 11637"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 76719.7971085017,
            "unit": "iter/sec",
            "range": "stddev: 0.000005149973354768671",
            "extra": "mean: 13.034445315147801 usec\nrounds: 23480"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.730555579604964,
            "unit": "iter/sec",
            "range": "stddev: 0.0004231735024204205",
            "extra": "mean: 43.993645315790346 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2862.158988872217,
            "unit": "iter/sec",
            "range": "stddev: 0.000017131179767858496",
            "extra": "mean: 349.38660077511355 usec\nrounds: 2322"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1876.1555249033931,
            "unit": "iter/sec",
            "range": "stddev: 0.0000304161116728176",
            "extra": "mean: 533.0048531299088 usec\nrounds: 1246"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1899.5884807205828,
            "unit": "iter/sec",
            "range": "stddev: 0.00002580885513989422",
            "extra": "mean: 526.4298084291729 usec\nrounds: 1044"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4144.727435788022,
            "unit": "iter/sec",
            "range": "stddev: 0.000015094131904304993",
            "extra": "mean: 241.270388823499 usec\nrounds: 3400"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 130.34938833150795,
            "unit": "iter/sec",
            "range": "stddev: 0.00019533311059082391",
            "extra": "mean: 7.671689240740999 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 26004.666392146966,
            "unit": "iter/sec",
            "range": "stddev: 0.000004496515759197608",
            "extra": "mean: 38.45463675327077 usec\nrounds: 18899"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 48408.9116425874,
            "unit": "iter/sec",
            "range": "stddev: 0.000007395729628564179",
            "extra": "mean: 20.657353492745266 usec\nrounds: 13972"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 40255.16582125169,
            "unit": "iter/sec",
            "range": "stddev: 0.000008119629135561362",
            "extra": "mean: 24.841532250553428 usec\nrounds: 12868"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 46856.38794333042,
            "unit": "iter/sec",
            "range": "stddev: 0.000007314300727119564",
            "extra": "mean: 21.341807251754688 usec\nrounds: 12659"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "46fc6d78c1579d395fc1cabdbad58ee101fce419",
          "message": "test(detection): add parametrized tests for elementwise handlers (#128)\n\n* test(detection): add parametrized tests for elementwise handlers\n\nAdd `numerical_jacobian_sparsity` helper in `tests/_utils.py` that\ncomputes expected sparsity from `jax.jacobian` using `np.isclose`.\n\nAdd 38 new tests covering:\n- Unary ops: `neg`, `exp`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,\n  `arctan`, `arcsinh`, `log1p`, `expm1`, `cbrt`, `exp2`, `sigmoid`,\n  `square`, `erf`, `log`, `sqrt`, `rsqrt`, `lgamma`, `digamma`,\n  `arcsin`, `arccos`, `arctanh`, `erfinv`, `arccosh`, `abs`, `conj`,\n  `real`, `imag`, `copy`, `i0e`, `i1e`\n- Zero-derivative ops: `floor`, `ceil`, `sign`, `round`\n- Binary ops: `maximum`, `minimum`, `power`, `arctan2`, `remainder`\n\nDocument analytical derivatives inline in both tests and dispatch code.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(tests): extract sparsity assertion helpers to _utils.py\n\nAdd `assert_jacobian_sparsity_exact` and `assert_jacobian_sparsity_conservative`\nhelpers to reduce boilerplate in elementwise handler tests.\nReduce test array sizes from 12 to 4 elements.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* refactor(tests): simplify sparsity assertion helpers\n\nRemove unnecessary input_proto and nnz_per_row parameters.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* test(detection): add binary op tests with mixed active/constant args\n\nTest combinations: first arg active with second constant, and vice versa.\nSeparate exact tests (power, arctan2) from conservative tests (max, min).\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-05-15T16:08:12+02:00",
          "tree_id": "e76303bb5319da49ea27980a06ff1c1067dbe3e1",
          "url": "https://github.com/adrhill/asdex/commit/46fc6d78c1579d395fc1cabdbad58ee101fce419"
        },
        "date": 1778854132456,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 1071.4796433294969,
            "unit": "iter/sec",
            "range": "stddev: 0.0000653632498152028",
            "extra": "mean: 933.2888461535468 usec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 27186.640664651106,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027969496522446133",
            "extra": "mean: 36.782771815578904 usec\nrounds: 7529"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 93124.3834572992,
            "unit": "iter/sec",
            "range": "stddev: 0.000001257136013105864",
            "extra": "mean: 10.738326127641265 usec\nrounds: 28866"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_value_and_materialization",
            "value": 65601.08788526806,
            "unit": "iter/sec",
            "range": "stddev: 0.0000016746313035496693",
            "extra": "mean: 15.243649644178667 usec\nrounds: 12787"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 90115.29278594503,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011610426731737024",
            "extra": "mean: 11.096895644286986 usec\nrounds: 30971"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.46949137990627,
            "unit": "iter/sec",
            "range": "stddev: 0.001013531832956669",
            "extra": "mean: 44.504790210528185 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 2875.5673551373648,
            "unit": "iter/sec",
            "range": "stddev: 0.000010324160005967651",
            "extra": "mean: 347.75746018031646 usec\nrounds: 2549"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 1906.0997406944407,
            "unit": "iter/sec",
            "range": "stddev: 0.000017648026833774072",
            "extra": "mean: 524.6315177796911 usec\nrounds: 1153"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_value_and_materialization",
            "value": 1911.0415096765064,
            "unit": "iter/sec",
            "range": "stddev: 0.000016067984854754363",
            "extra": "mean: 523.274871286954 usec\nrounds: 1111"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 4520.60746340112,
            "unit": "iter/sec",
            "range": "stddev: 0.000020681264173231105",
            "extra": "mean: 221.20920873931422 usec\nrounds: 3181"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 151.28982735836763,
            "unit": "iter/sec",
            "range": "stddev: 0.00010829475992701875",
            "extra": "mean: 6.6098297384612055 msec\nrounds: 65"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 27463.321380741316,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024281334681237705",
            "extra": "mean: 36.4122017922148 usec\nrounds: 17632"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 62059.155962255565,
            "unit": "iter/sec",
            "range": "stddev: 0.000003718119709359413",
            "extra": "mean: 16.113657759190296 usec\nrounds: 15717"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_value_and_materialization",
            "value": 54749.21136542341,
            "unit": "iter/sec",
            "range": "stddev: 0.000003948785363828489",
            "extra": "mean: 18.265103278392516 usec\nrounds: 11164"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 64884.764663199734,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029210218575422155",
            "extra": "mean: 15.411938460295648 usec\nrounds: 14186"
          }
        ]
      }
    ]
  }
}