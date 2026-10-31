# Research Paper Figures & Visual Artifacts

This directory contains all the publication-ready figures, architecture block diagrams, performance plots, ablation charts, and visual samples for the research paper on **Engineering Drawing Intelligence (EDI)**.

All newly generated charts are rendered at publication standard **300 DPI**.

---

## Complete Figure Catalog

| Filename | Figure Title | Description & Research Finding |
| :--- | :--- | :--- |
| [`fig01_rag_system_concept.png`](fig01_rag_system_concept.png) | Tri-Modal RAG Architecture | High-level conceptual overview showing how graph queries fuse with visual crops and document specs. |
| [`fig02_knowledge_graph_traversal.png`](fig02_knowledge_graph_traversal.png) | Graph Schema & Multi-Hop Traversal | Multi-hop topological graph rendering illustrating `CONNECTED_TO`, `CONTROLS`, and `FEEDS_TO`. |
| [`fig03_topological_graph_network.png`](fig03_topological_graph_network.png) | Physical Extracted Network | NetworkX layout of connected equipment and line networks extracted from a single P&ID sheet. |
| [`fig04_sample_pid_sheet_input.png`](fig04_sample_pid_sheet_input.png) | High-Resolution Input Sheet | Native scanned P&ID drawing (7168x4562 px) demonstrating micro-scale symbol challenges. |
| [`fig05_sheet_graph_overlay.png`](fig05_sheet_graph_overlay.png) | Visual Grounding & Graph Overlay | The extracted Knowledge Graph visually overlaid directly on the source drawing pixels. |
| [`fig06_pid_symbol_classes_atlas.png`](fig06_pid_symbol_classes_atlas.png) | P&ID Symbol Ontology Atlas | Visual dictionary of 32 classes (pumps, compressors, valves, vessels, instruments). |
| [`fig07_ocr_failure_analysis_contact_sheet.png`](fig07_ocr_failure_analysis_contact_sheet.png) | OCR Error Diagnostic Contact Sheet | Visual proof that vertical tags and bounding box clipping caused the initial value-tag accuracy drop. |
| [`fig08_yolo_labels_correlogram.jpg`](fig08_yolo_labels_correlogram.jpg) | Bounding Box Correlogram | Spatial distributions ($x, y, w, h$) of micro-symbols demonstrating severe class aspect ratio variance. |
| [`fig09_yolo_train_batch_samples.jpg`](fig09_yolo_train_batch_samples.jpg) | YOLO Training Tile Batches | Visual inspection of ground-truth bounding box annotations on 512x512 px tiles. |
| [`fig10_real_offshore_drawing_pfd.png`](fig10_real_offshore_drawing_pfd.png) | Real Industrial PFD Ingestion | Alpha Field Development Gas Treatment sheet: 15/15 equipment extracted with zero-shot Blueprint mode. |
| [`fig11_real_offshore_production_pid.png`](fig11_real_offshore_production_pid.png) | Dense Industrial P&ID Ingestion | LP Compressor Suction Cooler production sheet: 81 items, 93 pipe connections, 14 control loops. |
| [`fig12_rig_builder_procedural_pid.png`](fig12_rig_builder_procedural_pid.png) | Procedural Rig Builder P&ID | Full 7000x4600 px drawing generated from natural language spec, serving as ground-truth for flow direction. |
| [`fig13_complete_system_architecture.png`](fig13_complete_system_architecture.png) | End-to-End System Architecture | **Main Paper Architecture Diagram**: Details all 6 offline ingestion stages + 4 online QA stages. |
| [`fig14_detection_metrics_per_sheet.png`](fig14_detection_metrics_per_sheet.png) | YOLO Detection Performance per Sheet | Precision (99.09%), Recall (99.35%), and F1 (99.22%) across 20 unseen test sheets (2,311 symbols). |
| [`fig15_pidqa_benchmark_accuracies.png`](fig15_pidqa_benchmark_accuracies.png) | PIDQA Benchmark Accuracies | End-to-End QA results across 2,176 questions comparing the trained detector against the oracle ceiling. |
| [`fig16_threshold_tuning_ablation.png`](fig16_threshold_tuning_ablation.png) | NMS IoU & Conf Tuning Ablation | Demonstrates how tuning Conf (0.88) and NMS IoU (0.20) lifted full-sheet F1 from 82.0% to 99.22%. |
| [`fig17_pipeline_latency_breakdown.png`](fig17_pipeline_latency_breakdown.png) | Pipeline Latency Breakdown | Log-scale execution timing per stage running on Apple M3 Pro edge hardware. |

---

## LaTeX Snippets for Easy Insertion

### Architecture Figure:
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{paper_figures/fig13_complete_system_architecture.png}
\caption{End-to-end system architecture of Engineering Drawing Intelligence (EDI), showing the six-stage ingestion pipeline converting raster P\&IDs to an embedded knowledge graph (top) and the tri-modal evidence-grounded QA engine (bottom).}
\label{fig:architecture}
\end{figure*}
```

### Benchmark Results Figure:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{paper_figures/fig15_pidqa_benchmark_accuracies.png}
\caption{Evaluation on the PIDQA benchmark (2,176 questions across 17 sheets). The trained detector achieves an overall accuracy of 80.38\% with 0 query execution errors compared to an oracle ceiling of 86.12\%.}
\label{fig:benchmark}
\end{figure*}
```

### Detection Performance Figure:
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{paper_figures/fig14_detection_metrics_per_sheet.png}
\caption{Micro-symbol detection performance across 20 unseen evaluation sheets containing 2,311 symbols using YOLOv8s with tuned post-processing (overall F1: 99.22\%).}
\label{fig:detection}
\end{figure*}
```
