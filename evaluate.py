# # # =============================================================================
# # # evaluate.py — Evaluate and compare three LLM configurations
# # # Usage: python evaluate.py --samples 10
# # # Compares: Base Model vs Fine-tuned vs Fine-tuned + RAG
# # # =============================================================================

# # import argparse
# # import json
# # import time
# # import os
# # import pickle
# # import numpy as np
# # import datetime
# # import psutil
# # import os
# # import sys
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # from data.preprocess    import preprocess, get_normal_data
# # from ids.lightgbm_clf  import load_model as lgbm_load, predict as lgbm_predict
# # from ids.autoencoder   import load_model as ae_load,   predict as ae_predict
# # from ids.alert         import build_alert
# # from agent.tools       import (get_attack_details, lookup_mitre, search_cve,
# #                                 estimate_severity, suggest_mitigation)
# # import config

# # REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


# # # =============================================================================
# # # Scoring functions
# # # =============================================================================

# # def score_mitre(response: str, attack_type: str) -> float:
# #     """Check if correct MITRE technique ID is mentioned."""
# #     mitre   = lookup_mitre(attack_type)
# #     tech_id = mitre.get("technique", {}).get("id", "T0000")
# #     if tech_id == "T0000":
# #         return 0.5   # unknown technique — partial credit
# #     return 1.0 if tech_id in response else 0.0


# # def score_severity(response: str, expected_severity: str) -> float:
# #     """Check if correct severity level is mentioned."""
# #     return 1.0 if expected_severity.upper() in response.upper() else 0.0


# # def score_completeness(response: str) -> float:
# #     """Check if all 5 report sections are present."""
# #     sections = [
# #         ["attack", "summary"],
# #         ["mitre", "technique", "t1"],
# #         ["cve", "vulnerabilit"],
# #         ["severity", "cvss"],
# #         ["mitigation", "remediat", "recommend"],
# #     ]
# #     found = 0
# #     r = response.lower()
# #     for keywords in sections:
# #         if any(k in r for k in keywords):
# #             found += 1
# #     return found / len(sections)


# # def score_mitigation(response: str, attack_type: str) -> float:
# #     """Check if relevant mitigation keywords are present."""
# #     mit    = suggest_mitigation(attack_type)
# #     steps  = mit.get("mitigation_steps", [])
# #     if not steps:
# #         return 0.5
# #     # Extract key words from first 3 steps
# #     keywords = []
# #     for step in steps[:3]:
# #         words = [w.lower() for w in step.split() if len(w) > 4]
# #         keywords.extend(words[:3])
# #     if not keywords:
# #         return 0.5
# #     r     = response.lower()
# #     found = sum(1 for k in keywords if k in r)
# #     return min(found / len(keywords), 1.0)


# # def score_cve(response: str) -> float:
# #     """Check if CVE references are present."""
# #     r = response.lower()
# #     if "cve-" in r:
# #         return 1.0
# #     if "no cve" in r or "none" in r or "not found" in r:
# #         return 0.5   # correctly stated no CVEs
# #     return 0.0


# # def score_response_length(response: str) -> float:
# #     """Score based on response length — longer = more detailed."""
# #     length = len(response)
# #     if length >= 800:
# #         return 1.0
# #     elif length >= 500:
# #         return 0.75
# #     elif length >= 300:
# #         return 0.5
# #     elif length >= 100:
# #         return 0.25
# #     return 0.0


# # def score_response(response: str, alert: dict) -> dict:
# #     """Compute all scores for one LLM response."""
# #     attack_type = alert.get("attack_type", "unknown")
# #     severity    = alert.get("severity", "CRITICAL")

# #     scores = {
# #         "mitre_accuracy":    score_mitre(response, attack_type),
# #         "severity_correct":  score_severity(response, severity),
# #         "completeness":      score_completeness(response),
# #         "mitigation_quality":score_mitigation(response, attack_type),
# #         "cve_coverage":      score_cve(response),
# #         "response_length":   score_response_length(response),
# #     }
# #     scores["overall"] = round(sum(scores.values()) / len(scores), 3)
# #     return scores


# # # =============================================================================
# # # LLM caller — supports all three configurations
# # # =============================================================================

# # def call_llm(alert: dict, model_name: str, use_rag: bool = False) -> tuple:
# #     """
# #     Call LLM with given model and RAG setting.
# #     Returns: (response_text, elapsed_seconds)
# #     """
# #     import requests

# #     # Gather tool results
# #     attack_type  = alert.get("attack_type", "unknown")
# #     tool_results = {
# #         "attack_details": get_attack_details(attack_type),
# #         "mitre":          lookup_mitre(attack_type),
# #         "cve":            search_cve(attack_type),
# #         "severity":       estimate_severity(
# #                               alert.get("anomaly_score", 0),
# #                               alert.get("lgbm_confidence", 0),
# #                               alert.get("anomaly_threshold", 1)
# #                           ),
# #         "mitigation":     suggest_mitigation(attack_type),
# #     }

# #     # RAG context
# #     rag_block = ""
# #     if use_rag:
# #         try:
# #             from rag.retriever import retrieve
# #             query     = f"{attack_type} IoT attack mitigation"
# #             rag_block = retrieve(query, top_k=1)  # reduced from 2 to 1
# #         except Exception as e:
# #             rag_block = ""

# #     # Build prompt
# #     mitre      = tool_results["mitre"].get("technique", {})
# #     cves       = tool_results["cve"].get("results", [])
# #     severity   = tool_results["severity"]
# #     mitigation = tool_results["mitigation"].get("mitigation_steps", [])
# #     details    = tool_results["attack_details"].get("details", "Unknown.")

# #     cve_text = "; ".join([
# #         f"{c.get('cve_id','N/A')}(CVSS:{c.get('cvss_score','N/A')})"
# #         for c in cves[:2]
# #     ]) or "None found"

# #     mit_text = " | ".join(mitigation[:3])

# #     user = f"""ALERT:
# # Attack : {alert['attack_type']}
# # LightGBM: {alert['lgbm_label']} ({alert['lgbm_confidence']*100:.0f}%)
# # Anomaly : {alert['is_anomaly']} (score:{alert['anomaly_score']})
# # Severity: {alert['severity']}
# # {rag_block}
# # INTELLIGENCE:
# # Details : {details[:200]}
# # MITRE   : {mitre.get('id','N/A')} - {mitre.get('name','N/A')} ({mitre.get('tactic','N/A')})
# # CVEs    : {cve_text}
# # Severity: {severity.get('severity','N/A')} CVSS:{severity.get('cvss_estimate','N/A')}
# # Mitigation: {mit_text}

# # Write a structured security report:
# # 1) Attack Summary 2) MITRE Mapping 3) CVEs 4) Severity 5) Mitigation Steps"""

# #     messages = [
# #         {"role": "system", "content": "/no_think\nYou are an IoT security analyst. Write a clear structured security report."},
# #         {"role": "user",   "content": user},
# #     ]
    
# #     ctx_size = 1024 if use_rag else 512
# #     payload = {
# #         "model":   model_name,
# #         "messages": messages,
# #         "stream":  False,
# #         "options": {"num_ctx": ctx_size, "temperature": 0.3, "num_predict": 1024} #earlier num_ctx= 2048
# #     }

# #     # start = time.time()
# #     # try:
# #     #     resp    = requests.post(f"{config.LLM_HOST}/api/chat",
# #     #                              json=payload, timeout=config.LLM_TIMEOUT)
# #     #     resp.raise_for_status()
# #     #     answer  = resp.json().get("message", {}).get("content", "No response.")
# #     process = psutil.Process(os.getpid())
# #     ram_before = process.memory_info().rss / (1024 ** 2)  # MB
# #     prompt_tokens = 0
# #     response_tokens = 0
# #     total_tokens = 0

# #     start = time.time()
# #     try:
# #         resp    = requests.post(f"{config.LLM_HOST}/api/chat",
# #                                   json=payload, timeout=config.LLM_TIMEOUT)
# #         raw_resp = resp.json()
# #         answer  = raw_resp.get("message", {}).get("content", "No response.")
    
# #         # Extract token counts from Ollama response
# #         prompt_tokens   = raw_resp.get("prompt_eval_count", 0)
# #         response_tokens = raw_resp.get("eval_count", 0)
# #         total_tokens    = prompt_tokens + response_tokens


# #         if "</think>" in answer:
# #             answer = answer.split("</think>")[-1].strip()
# #     except Exception as e:
# #         answer = f"Error: {str(e)}"

# #     elapsed = round(time.time() - start, 2)
# #     ram_after  = process.memory_info().rss / (1024 ** 2)
# #     ram_used   = round(ram_after - ram_before, 2)
# #     # return answer, elapsed
# #     return answer, elapsed, {
# #     "prompt_tokens":   prompt_tokens,
# #     "response_tokens": response_tokens,
# #     "total_tokens":    total_tokens,
# #     "ram_used_mb":     ram_used,
# # }


# # # =============================================================================
# # # Main evaluation loop
# # # =============================================================================

# # def evaluate(n_samples: int = 10,
# #              base_model: str = "qwen3:4b",
# #              ft_model:   str = "iot-ids-llm"):
# #     """
# #     Run evaluation across all three configurations.
# #     """
# #     print("\n" + "="*65)
# #     print("EVALUATOR — Base vs Fine-tuned vs Fine-tuned + RAG")
# #     print("="*65)

# #     # ── Load IDS models ───────────────────────────────────────────────────────
# #     print("\n[Eval] Loading IDS models...")
# #     with open(os.path.join(config.MODEL_DIR, "label_encoder.pkl"), "rb") as f:
# #         le = pickle.load(f)
# #     with open(os.path.join(config.MODEL_DIR, "feature_names.pkl"), "rb") as f:
# #         feature_names = pickle.load(f)

# #     lgbm_model             = lgbm_load()
# #     ae_model, ae_threshold = ae_load()

# #     # ── Load test data ─────────────────────────────────────────────────────────
# #     _, X_test, _, y_test, _, _ = preprocess()

# #     # Sample mixed flows (50% benign, 50% attack)
# #     benign_idx     = le.transform(["Benign"])[0] if "Benign" in le.classes_ else -1
# #     benign_indices = np.where(y_test == benign_idx)[0] if benign_idx >= 0 else np.array([])
# #     attack_indices = np.where(y_test != benign_idx)[0]

# #     n_each  = n_samples // 2
# #     #sampled = []
# #     sampled = []
# #     if n_samples == 1:
# #         # Single sample — pick one attack flow directly
# #         sampled.extend(np.random.choice(attack_indices, 1, replace=False).tolist())
# #     else:
# #         n_each = max(1, n_samples // 2)
# #         if len(benign_indices) > 0:
# #             sampled.extend(np.random.choice(benign_indices,
# #                         min(n_each, len(benign_indices)), replace=False).tolist())
# #         sampled.extend(np.random.choice(attack_indices,
# #                     min(n_each, len(attack_indices)), replace=False).tolist())
# #     indices = np.random.permutation(sampled)
# #     """if len(benign_indices) > 0:
# #         sampled.extend(np.random.choice(benign_indices,
# #                         min(n_each, len(benign_indices)), replace=False).tolist())
# #     sampled.extend(np.random.choice(attack_indices,
# #                     min(n_each, len(attack_indices)), replace=False).tolist())
# #     indices = np.random.permutation(sampled)"""

# #     # ── Configurations ────────────────────────────────────────────────────────
# #     configs = [
# #         {"name": "Base Model",          "model": base_model, "rag": False},
# #         {"name": "Fine-tuned",          "model": ft_model,   "rag": False},
# #         {"name": "Fine-tuned + RAG",    "model": ft_model,   "rag": True},
# #     ]

# #     # ── Results storage ───────────────────────────────────────────────────────
# #     results = {c["name"]: {
# #         "scores":   [],
# #         "times":    [],
# #         "overhead":  [],
# #         "responses": []
# #     } for c in configs}

# #     alerts_evaluated = 0

# #     # ── Evaluation loop ───────────────────────────────────────────────────────
# #     for i, idx in enumerate(indices):
# #         features    = X_test[idx]
# #         lgbm_result = lgbm_predict(lgbm_model, features, le)
# #         ae_result   = ae_predict(ae_model, ae_threshold, features)
# #         alert       = build_alert(features, feature_names, lgbm_result, ae_result)

# #         if alert is None:
# #             continue

# #         alerts_evaluated += 1
# #         print(f"\n[Eval] Alert {alerts_evaluated}/{len(indices)}: "
# #               f"{alert['attack_type']} | {alert['severity']}")

# #         for cfg in configs:
# #             print(f"  → Testing: {cfg['name']}...")
# #             # response, elapsed = call_llm(alert, cfg["model"], cfg["rag"])
# #             response, elapsed, overhead = call_llm(alert, cfg["model"], cfg["rag"])
# #             scores            = score_response(response, alert)

# #             results[cfg["name"]]["scores"].append(scores)
# #             results[cfg["name"]]["times"].append(elapsed)
# #             results[cfg["name"]]["overhead"].append(overhead)
# #             results[cfg["name"]]["responses"].append({
# #                 "alert":    alert["attack_type"],
# #                 "response": response[:500],
# #                 "scores":   scores,
# #                 "time":     elapsed
# #             })

# #             print(f"     Score: {scores['overall']*100:.0f}% | "
# #                   f"Time: {elapsed}s")

# #     # ── Compute averages ──────────────────────────────────────────────────────
# #     summary = {}
# #     for cfg in configs:
# #         name   = cfg["name"]
# #         scores = results[name]["scores"]
# #         times  = results[name]["times"]

# #         if not scores:
# #             continue

# #         avg_scores = {}
# #         for metric in scores[0].keys():
# #             avg_scores[metric] = round(
# #                 sum(s[metric] for s in scores) / len(scores), 3
# #             )
# #         avg_scores["avg_response_time"] = round(
# #             sum(times) / len(times), 2
# #         )
# #         summary[name] = avg_scores

# #     # ── Print comparison table ────────────────────────────────────────────────
# #     print("\n\n" + "="*65)
# #     print("EVALUATION RESULTS")
# #     print("="*65)

# #     metrics = [
# #         ("mitre_accuracy",     "MITRE Accuracy"),
# #         ("severity_correct",   "Severity Correct"),
# #         ("completeness",       "Completeness"),
# #         ("mitigation_quality", "Mitigation Quality"),
# #         ("cve_coverage",       "CVE Coverage"),
# #         ("response_length",    "Response Detail"),
# #         ("overall",            "Overall Score"),
# #         ("avg_response_time",  "Avg Response Time (s)"),
# #     ]

# #     col_w = 22
# #     header = f"{'Metric':<25}" + "".join(f"{c['name']:<{col_w}}" for c in configs)
# #     print(header)
# #     print("-" * (25 + col_w * len(configs)))

# #     for key, label in metrics:
# #         row = f"{label:<25}"
# #         for cfg in configs:
# #             name = cfg["name"]
# #             val  = summary.get(name, {}).get(key, 0)
# #             if key == "avg_response_time":
# #                 row += f"{val:<{col_w}.1f}"
# #             else:
# #                 row += f"{val*100:<{col_w}.1f}%"
# #         print(row)

# #     print("="*65)
# #     print("\n── Computational Overhead ──────────────────────────────────")
# #     oh_metrics = [
# #         ("prompt_tokens",   "Prompt Tokens"),
# #         ("response_tokens", "Response Tokens"),
# #         ("total_tokens",    "Total Tokens"),
# #         ("ram_used_mb",     "RAM Used (MB)"),
# # ]
# #     header = f"{'Metric':<25}" + "".join(f"{c['name']:<{col_w}}" for c in configs)
# #     print(header)
# #     print("-" * (25 + col_w * len(configs)))
# #     for key, label in oh_metrics:
# #         row = f"{label:<25}"
# #     for cfg in configs:
# #         name = cfg["name"]
# #         oh   = results[name]["overhead"]
# #         avg  = round(sum(o.get(key, 0) for o in oh) / max(len(oh), 1), 1)
# #         row += f"{avg:<{col_w}}"
# #     print(row)

# # # Model size info
# #     print(f"\n── Model Size on Disk ──────────────────────────────────────")
# #     print(f"  Base model  (qwen3:4b Q4_K_M GGUF) : ~2.5 GB")
# #     print(f"  Fine-tuned  (iot-ids-llm Q4_K_M)   : ~2.5 GB")
# #     print(f"  Suitable for edge deployment        : Yes (RPi 5 / Jetson Orin)")

# #     # ── Save reports ──────────────────────────────────────────────────────────
# #     os.makedirs(REPORT_DIR, exist_ok=True)
# #     ts        = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
# #     json_path = os.path.join(REPORT_DIR, f"eval_{ts}.json")
# #     md_path   = os.path.join(REPORT_DIR, f"eval_{ts}.md")

# #     # JSON
# #     with open(json_path, "w") as f:
# #         json.dump({"summary": summary, "details": results}, f, indent=2)

# #     # Markdown
# #     md = f"# Evaluation Report\n**Generated:** {ts} UTC\n\n"
# #     md += "## Summary\n\n"
# #     md += f"| Metric | Base Model | Fine-tuned | Fine-tuned + RAG |\n"
# #     md += f"|--------|-----------|------------|------------------|\n"
# #     for key, label in metrics:
# #         row = f"| {label} |"
# #         for cfg in configs:
# #             val = summary.get(cfg["name"], {}).get(key, 0)
# #             if key == "avg_response_time":
# #                 row += f" {val:.1f}s |"
# #             else:
# #                 row += f" {val*100:.1f}% |"
# #         md += row + "\n"

# #     md += "\n## Per-Alert Details\n\n"
# #     for cfg in configs:
# #         md += f"### {cfg['name']}\n\n"
# #         for item in results[cfg["name"]]["responses"]:
# #             md += f"**Alert:** {item['alert']} | "
# #             md += f"**Score:** {item['scores']['overall']*100:.0f}% | "
# #             md += f"**Time:** {item['time']}s\n\n"
# #             md += f"{item['response'][:300]}...\n\n---\n\n"

# #     with open(md_path, "w") as f:
# #         f.write(md)

# #     print(f"\n[Eval] Reports saved:")
# #     print(f"  JSON → {json_path}")
# #     print(f"  MD   → {md_path}")
# #     print(f"\n[Eval] Alerts evaluated: {alerts_evaluated}")


# # # =============================================================================
# # # Entry point
# # # =============================================================================

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="IoT IDS Evaluator")
# #     parser.add_argument("--samples",    type=int, default=10,
# #                         help="Number of test flows to evaluate")
# #     parser.add_argument("--base-model", type=str, default="qwen3:4b",
# #                         help="Base model name in Ollama")
# #     parser.add_argument("--ft-model",   type=str, default="iot-ids-llm",
# #                         help="Fine-tuned model name in Ollama")
# #     args = parser.parse_args()

# #     evaluate(
# #         n_samples  = args.samples,
# #         base_model = args.base_model,
# #         ft_model   = args.ft_model
# #     )


# # =============================================================================
# # evaluate.py — Compare Base Model vs Fine-tuned vs Fine-tuned + RAG
# # Includes: scoring, computational overhead, RAM, tokens, model size
# # Usage: python evaluate.py --samples 10
# # =============================================================================

# import argparse
# import json
# import time
# import os
# import pickle
# import numpy as np
# import datetime
# import sys
# import psutil
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from data.preprocess    import preprocess, get_normal_data
# from ids.lightgbm_clf  import load_model as lgbm_load, predict as lgbm_predict
# from ids.autoencoder   import load_model as ae_load,   predict as ae_predict
# from ids.alert         import build_alert
# from agent.tools       import (get_attack_details, lookup_mitre,
#                                 estimate_severity, suggest_mitigation)
# import config

# REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


# # =============================================================================
# # Scoring functions
# # =============================================================================

# def score_mitre(response: str, attack_type: str) -> float:
#     from agent.tools import lookup_mitre as lm
#     tech_id = lm(attack_type).get("technique", {}).get("id", "T0000")
#     if tech_id == "T0000":
#         return 0.5
#     return 1.0 if tech_id in response else 0.0


# def score_severity(response: str, expected_severity: str) -> float:
#     return 1.0 if expected_severity.upper() in response.upper() else 0.0


# def score_completeness(response: str) -> float:
#     sections = [
#         ["attack", "summary"],
#         ["mitre", "technique", "t1"],
#         ["cve", "vulnerabilit"],
#         ["severity", "cvss"],
#         ["mitigation", "remediat", "recommend"],
#     ]
#     r     = response.lower()
#     found = sum(1 for kws in sections if any(k in r for k in kws))
#     return found / len(sections)


# def score_mitigation(response: str, attack_type: str) -> float:
#     from agent.tools import suggest_mitigation as sm
#     steps    = sm(attack_type).get("mitigation_steps", [])
#     if not steps:
#         return 0.5
#     keywords = []
#     for step in steps[:3]:
#         keywords.extend([w.lower() for w in step.split() if len(w) > 4][:3])
#     if not keywords:
#         return 0.5
#     r = response.lower()
#     return min(sum(1 for k in keywords if k in r) / len(keywords), 1.0)


# def score_cve(response: str) -> float:
#     r = response.lower()
#     if "cve-" in r:
#         return 1.0
#     if any(x in r for x in ["no cve", "none", "not found"]):
#         return 0.5
#     return 0.0


# def score_length(response: str) -> float:
#     n = len(response)
#     if n >= 800:  return 1.0
#     if n >= 500:  return 0.75
#     if n >= 300:  return 0.5
#     if n >= 100:  return 0.25
#     return 0.0


# def score_response(response: str, alert: dict) -> dict:
#     attack_type = alert.get("attack_type", "unknown")
#     severity    = alert.get("severity", "CRITICAL")
#     scores = {
#         "mitre_accuracy":     score_mitre(response, attack_type),
#         "severity_correct":   score_severity(response, severity),
#         "completeness":       score_completeness(response),
#         "mitigation_quality": score_mitigation(response, attack_type),
#         "cve_coverage":       score_cve(response),
#         "response_length":    score_length(response),
#     }
#     scores["overall"] = round(sum(scores.values()) / len(scores), 3)
#     return scores


# # =============================================================================
# # LLM caller
# # =============================================================================

# def call_llm(alert: dict, model_name: str, use_rag: bool = False) -> tuple:
#     """
#     Call LLM. Returns (response, elapsed, overhead_dict).
#     overhead_dict: prompt_tokens, response_tokens, total_tokens, ram_used_mb
#     """
#     import requests

#     attack_type  = alert.get("attack_type", "unknown")
#     tool_results = {
#         "attack_details": get_attack_details(attack_type),
#         "mitre":          lookup_mitre(attack_type),
#         "severity":       estimate_severity(
#                               alert.get("anomaly_score", 0),
#                               alert.get("lgbm_confidence", 0),
#                               alert.get("anomaly_threshold", 1),
#                               attack_type
#                           ),
#         "mitigation":     suggest_mitigation(attack_type),
#     }

#     rag_block = ""
#     if use_rag:
#         try:
#             from rag.retriever import retrieve
#             rag_block = retrieve(
#                 f"{attack_type} IoT attack mitigation", top_k=1
#             )[:100]
#         except Exception:
#             rag_block = ""

#     mitre      = tool_results["mitre"].get("technique", {})
#     severity   = tool_results["severity"]
#     mitigation = tool_results["mitigation"].get("mitigation_steps", [])
#     details    = tool_results["attack_details"].get("details", "Unknown.")
#     mit_text   = " | ".join(mitigation[:3])

#     user = f"""ALERT: {alert['attack_type']} | {alert['severity']}
# LightGBM: {alert['lgbm_label']} ({alert['lgbm_confidence']*100:.0f}%)
# Anomaly: {alert['is_anomaly']} (score:{alert['anomaly_score']})
# {rag_block}
# INTELLIGENCE:
# Details: {details[:150]}
# MITRE: {mitre.get('id','N/A')} - {mitre.get('name','N/A')} ({mitre.get('tactic','N/A')})
# CVSS: {severity.get('cvss_score','N/A')} ({severity.get('cvss_vector','N/A')})
# Mitigation: {mit_text[:100]}

# Write security report:
# 1) Attack Summary 2) MITRE Mapping 3) CVEs 4) Severity 5) Mitigation Steps"""
	
#     # ctx_size = 1024 if use_rag else 512
#     # messages = [
#     #     {"role": "system", "content": "/no_think\nYou are an IoT security analyst. Write a structured security report."},
#     #     {"role": "user",   "content": user},
#     # ]
#     messages = [
#     {"role": "system", "content": "You are an IoT security analyst. Write a structured security report. Be concise."},
#     {"role": "user",   "content": f"/no_think\n{user}"},
# 	]
#     payload = {
#         "model":    model_name,
#         "messages": messages,
#         "stream":   False,
#         "options":  {"num_ctx": 256, "temperature": 0.1, "num_predict": 128}
#         # "options":  {"num_ctx": ctx_size, "temperature": 0.3, "num_predict": 256}
#     }

#     # RAM before
#     process    = psutil.Process(os.getpid())
#     ram_before = process.memory_info().rss / (1024 ** 2)

#     start = time.time()
#     try:
#         resp     = requests.post(f"{config.LLM_HOST}/api/chat",
#                                   json=payload, timeout=config.LLM_TIMEOUT)
#         elapsed  = round(time.time() - start, 2)
#         raw      = resp.json()
#         answer   = raw.get("message", {}).get("content", "")
#         if "</think>" in answer:
#             stripped = answer.split("</think>")[-1].strip()
#             answer   = stripped if stripped else answer.split("</think>")[0].strip()

#         prompt_tokens   = raw.get("prompt_eval_count", 0)
#         response_tokens = raw.get("eval_count", 0)
#         total_tokens    = prompt_tokens + response_tokens

#     except Exception as e:
#         elapsed         = round(time.time() - start, 2)
#         answer          = f"Error: {str(e)}"
#         prompt_tokens   = 0
#         response_tokens = 0
#         total_tokens    = 0

#     # RAM after
#     ram_after = process.memory_info().rss / (1024 ** 2)
#     ram_used  = round(ram_after - ram_before, 2)

#     overhead = {
#         "prompt_tokens":   prompt_tokens,
#         "response_tokens": response_tokens,
#         "total_tokens":    total_tokens,
#         "ram_used_mb":     ram_used,
#     }
#     return answer, elapsed, overhead


# # =============================================================================
# # Main evaluation
# # =============================================================================

# def evaluate(n_samples: int = 10,
#              base_model: str = "qwen3:4b",
#              ft_model:   str = "iot-ids-llm"):

#     print("\n" + "="*65)
#     print("EVALUATOR — Base vs Fine-tuned vs Fine-tuned + RAG")
#     print("="*65)

#     # Load IDS models
#     print("\n[Eval] Loading IDS models...")
#     with open(os.path.join(config.MODEL_DIR, "label_encoder.pkl"), "rb") as f:
#         le = pickle.load(f)
#     with open(os.path.join(config.MODEL_DIR, "feature_names.pkl"), "rb") as f:
#         feature_names = pickle.load(f)

#     lgbm_model             = lgbm_load()
#     ae_model, ae_threshold = ae_load()
#     _, X_test, _, y_test, _, _ = preprocess()

#     # Sample mixed flows
#     benign_idx     = le.transform(["Benign"])[0] if "Benign" in le.classes_ else -1
#     benign_indices = np.where(y_test == benign_idx)[0] if benign_idx >= 0 else np.array([])
#     attack_indices = np.where(y_test != benign_idx)[0]

#     sampled = []
#     if n_samples == 1:
#         sampled = np.random.choice(attack_indices, 1, replace=False).tolist()
#     else:
#         n_each = max(1, n_samples // 2)
#         if len(benign_indices) > 0:
#             sampled.extend(np.random.choice(benign_indices,
#                             min(n_each, len(benign_indices)), replace=False).tolist())
#         sampled.extend(np.random.choice(attack_indices,
#                         min(n_each, len(attack_indices)), replace=False).tolist())

#     indices = np.random.permutation(sampled)

#     configs = [
#         {"name": "Base Model",       "model": base_model, "rag": False},
#         {"name": "Fine-tuned",       "model": ft_model,   "rag": False},
#         {"name": "Fine-tuned + RAG", "model": ft_model,   "rag": True},
#     ]

#     results = {c["name"]: {
#         "scores":    [],
#         "times":     [],
#         "overhead":  [],
#         "responses": []
#     } for c in configs}

#     alerts_evaluated = 0

#     for i, idx in enumerate(indices):
#         features    = X_test[idx]
#         lgbm_result = lgbm_predict(lgbm_model, features, le)
#         ae_result   = ae_predict(ae_model, ae_threshold, features)
#         alert       = build_alert(features, feature_names, lgbm_result, ae_result)
#         if alert is None:
#             continue

#         alerts_evaluated += 1
#         print(f"\n[Eval] Alert {alerts_evaluated}: "
#               f"{alert['attack_type']} | {alert['severity']}")

#         for cfg in configs:
#             print(f"  → {cfg['name']}...")
#             response, elapsed, overhead = call_llm(alert, cfg["model"], cfg["rag"])
#             scores = score_response(response, alert)

#             results[cfg["name"]]["scores"].append(scores)
#             results[cfg["name"]]["times"].append(elapsed)
#             results[cfg["name"]]["overhead"].append(overhead)
#             results[cfg["name"]]["responses"].append({
#                 "alert":    alert["attack_type"],
#                 "response": response[:500],
#                 "scores":   scores,
#                 "time":     elapsed,
#                 "overhead": overhead,
#             })
#             print(f"     Score: {scores['overall']*100:.0f}% | "
#                   f"Time: {elapsed}s | "
#                   f"Tokens: {overhead['total_tokens']} | "
#                   f"RAM: {overhead['ram_used_mb']}MB")

#     # ── Compute averages ──────────────────────────────────────────────────────
#     summary = {}
#     for cfg in configs:
#         name   = cfg["name"]
#         scores = results[name]["scores"]
#         times  = results[name]["times"]
#         ohs    = results[name]["overhead"]
#         if not scores:
#             continue
#         avg_scores = {}
#         for metric in scores[0].keys():
#             avg_scores[metric] = round(
#                 sum(s[metric] for s in scores) / len(scores), 3)
#         avg_scores["avg_response_time_s"] = round(sum(times) / len(times), 2)
#         avg_scores["avg_prompt_tokens"]   = round(sum(o["prompt_tokens"]   for o in ohs) / len(ohs), 1)
#         avg_scores["avg_response_tokens"] = round(sum(o["response_tokens"] for o in ohs) / len(ohs), 1)
#         avg_scores["avg_total_tokens"]    = round(sum(o["total_tokens"]    for o in ohs) / len(ohs), 1)
#         avg_scores["avg_ram_used_mb"]     = round(sum(o["ram_used_mb"]     for o in ohs) / len(ohs), 2)
#         summary[name] = avg_scores

#     col_w = 22

#     # ── Quality scores table ──────────────────────────────────────────────────
#     print("\n\n" + "="*65)
#     print("EVALUATION RESULTS — Quality Scores")
#     print("="*65)

#     quality_metrics = [
#         ("mitre_accuracy",     "MITRE Accuracy"),
#         ("severity_correct",   "Severity Correct"),
#         ("completeness",       "Completeness"),
#         ("mitigation_quality", "Mitigation Quality"),
#         ("cve_coverage",       "CVE Coverage"),
#         ("response_length",    "Response Detail"),
#         ("overall",            "Overall Score"),
#     ]

#     header = f"{'Metric':<25}" + "".join(f"{c['name']:<{col_w}}" for c in configs)
#     print(header)
#     print("-" * (25 + col_w * len(configs)))
#     for key, label in quality_metrics:
#         row = f"{label:<25}"
#         for cfg in configs:
#             val = summary.get(cfg["name"], {}).get(key, 0)
#             row += f"{val*100:<{col_w}.1f}%"
#         print(row)
#     print("="*65)

#     # ── Computational overhead table ──────────────────────────────────────────
#     print("\n" + "="*65)
#     print("COMPUTATIONAL OVERHEAD")
#     print("="*65)

#     overhead_metrics = [
#         ("avg_response_time_s",  "Avg Response Time (s)"),
#         ("avg_prompt_tokens",    "Avg Prompt Tokens"),
#         ("avg_response_tokens",  "Avg Response Tokens"),
#         ("avg_total_tokens",     "Avg Total Tokens"),
#         ("avg_ram_used_mb",      "Avg RAM Used (MB)"),
#     ]

#     header = f"{'Metric':<25}" + "".join(f"{c['name']:<{col_w}}" for c in configs)
#     print(header)
#     print("-" * (25 + col_w * len(configs)))
#     for key, label in overhead_metrics:
#         row = f"{label:<25}"
#         for cfg in configs:
#             val = summary.get(cfg["name"], {}).get(key, 0)
#             row += f"{val:<{col_w}.2f}"
#         print(row)
#     print("="*65)

#     # ── Model size table ──────────────────────────────────────────────────────
#     print("\n" + "="*65)
#     print("MODEL SIZE ON DISK")
#     print("="*65)
#     print(f"{'Model':<30} {'Format':<15} {'Size':<10} {'Edge Deploy'}")
#     print("-"*65)
#     print(f"{'qwen3:4b (Base)':<30} {'Q4_K_M GGUF':<15} {'~2.5 GB':<10} Yes (RPi5 / Jetson)")
#     print(f"{'iot-ids-llm (Fine-tuned)':<30} {'Q4_K_M GGUF':<15} {'~2.5 GB':<10} Yes (RPi5 / Jetson)")
#     print(f"{'all-MiniLM-L6-v2 (RAG)':<30} {'PyTorch':<15} {'~90 MB':<10} Yes")
#     print(f"{'FAISS Index':<30} {'Binary':<15} {'~varies':<10} Yes")
#     print("="*65)

#     # ── Save reports ──────────────────────────────────────────────────────────
#     os.makedirs(REPORT_DIR, exist_ok=True)
#     ts        = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#     json_path = os.path.join(REPORT_DIR, f"eval_{ts}.json")
#     md_path   = os.path.join(REPORT_DIR, f"eval_{ts}.md")

#     with open(json_path, "w") as f:
#         json.dump({"summary": summary, "details": results}, f, indent=2)

#     # Markdown report
#     md  = f"# Evaluation Report\n**Generated:** {ts} UTC\n\n"
#     md += "## Quality Scores\n\n"
#     md += f"| Metric | Base Model | Fine-tuned | Fine-tuned + RAG |\n"
#     md += f"|--------|-----------|------------|------------------|\n"
#     for key, label in quality_metrics:
#         row = f"| {label} |"
#         for cfg in configs:
#             val = summary.get(cfg["name"], {}).get(key, 0)
#             row += f" {val*100:.1f}% |"
#         md += row + "\n"

#     md += "\n## Computational Overhead\n\n"
#     md += f"| Metric | Base Model | Fine-tuned | Fine-tuned + RAG |\n"
#     md += f"|--------|-----------|------------|------------------|\n"
#     for key, label in overhead_metrics:
#         row = f"| {label} |"
#         for cfg in configs:
#             val = summary.get(cfg["name"], {}).get(key, 0)
#             row += f" {val:.2f} |"
#         md += row + "\n"

#     md += "\n## Model Size on Disk\n\n"
#     md += "| Model | Format | Size | Edge Deployable |\n"
#     md += "|-------|--------|------|-----------------|\n"
#     md += "| qwen3:4b (Base) | Q4_K_M GGUF | ~2.5 GB | Yes |\n"
#     md += "| iot-ids-llm (Fine-tuned) | Q4_K_M GGUF | ~2.5 GB | Yes |\n"
#     md += "| all-MiniLM-L6-v2 (RAG embedder) | PyTorch | ~90 MB | Yes |\n"

#     md += "\n## Per-Alert Details\n\n"
#     for cfg in configs:
#         md += f"### {cfg['name']}\n\n"
#         for item in results[cfg["name"]]["responses"]:
#             md += f"**Alert:** {item['alert']} | "
#             md += f"**Score:** {item['scores']['overall']*100:.0f}% | "
#             md += f"**Time:** {item['time']}s | "
#             md += f"**Tokens:** {item['overhead']['total_tokens']}\n\n"
#             md += f"{item['response'][:300]}...\n\n---\n\n"

#     with open(md_path, "w") as f:
#         f.write(md)

#     print(f"\n[Eval] Reports saved:")
#     print(f"  JSON → {json_path}")
#     print(f"  MD   → {md_path}")
#     print(f"  Alerts evaluated: {alerts_evaluated}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="IoT IDS Evaluator")
#     parser.add_argument("--samples",    type=int, default=10)
#     parser.add_argument("--base-model", type=str, default="qwen3:4b")
#     parser.add_argument("--ft-model",   type=str, default="iot-ids-llm")
#     args = parser.parse_args()
#     evaluate(n_samples=args.samples, base_model=args.base_model, ft_model=args.ft_model)



# =============================================================================
# evaluate.py — Compare Base Model vs Fine-tuned vs Fine-tuned + RAG
# Scoring: same as original version
# Additional: overhead table + model size table (not in scoring)
# Usage: python evaluate.py --samples 10
# =============================================================================

import argparse
import json
import time
import os
import pickle
import numpy as np
import datetime
import sys
import psutil
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocess    import preprocess, get_normal_data
from ids.lightgbm_clf  import load_model as lgbm_load, predict as lgbm_predict
from ids.autoencoder   import load_model as ae_load,   predict as ae_predict
from ids.alert         import build_alert
from agent.tools       import (get_attack_details, lookup_mitre,
                                estimate_severity, suggest_mitigation)
import config

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


# =============================================================================
# Scoring functions — identical to original evaluate.py
# =============================================================================

def score_mitre(response: str, attack_type: str) -> float:
    """Check if correct MITRE technique ID is mentioned."""
    from agent.tools import lookup_mitre as lm
    tech_id = lm(attack_type).get("technique", {}).get("id", "T0000")
    if tech_id == "T0000":
        return 0.5
    return 1.0 if tech_id in response else 0.0


def score_severity(response: str, expected_severity: str) -> float:
    """Check if correct severity level is mentioned."""
    return 1.0 if expected_severity.upper() in response.upper() else 0.0


def score_completeness(response: str) -> float:
    """Check if all 5 report sections are present."""
    sections = [
        ["attack", "summary"],
        ["mitre", "technique", "t1"],
        ["cve", "vulnerabilit"],
        ["severity", "cvss"],
        ["mitigation", "remediat", "recommend"],
    ]
    r     = response.lower()
    found = sum(1 for kws in sections if any(k in r for k in kws))
    return found / len(sections)


def score_mitigation(response: str, attack_type: str) -> float:
    """Check if relevant mitigation keywords are present."""
    from agent.tools import suggest_mitigation as sm
    steps    = sm(attack_type).get("mitigation_steps", [])
    if not steps:
        return 0.5
    keywords = []
    for step in steps[:3]:
        keywords.extend([w.lower() for w in step.split() if len(w) > 4][:3])
    if not keywords:
        return 0.5
    r = response.lower()
    return min(sum(1 for k in keywords if k in r) / len(keywords), 1.0)


def score_cve(response: str) -> float:
    """Check if CVE references are present."""
    r = response.lower()
    if "cve-" in r:
        return 1.0
    if any(x in r for x in ["no cve", "none", "not found"]):
        return 0.5
    return 0.0


def score_length(response: str) -> float:
    """Score based on response length."""
    n = len(response)
    if n >= 800:  return 1.0
    if n >= 500:  return 0.75
    if n >= 300:  return 0.5
    if n >= 100:  return 0.25
    return 0.0


def score_response(response: str, alert: dict) -> dict:
    """Compute all scores for one LLM response."""
    attack_type = alert.get("attack_type", "unknown")
    severity    = alert.get("severity", "CRITICAL")
    scores = {
        "mitre_accuracy":     score_mitre(response, attack_type),
        "severity_correct":   score_severity(response, severity),
        "completeness":       score_completeness(response),
        "mitigation_quality": score_mitigation(response, attack_type),
        "cve_coverage":       score_cve(response),
        "response_length":    score_length(response),
    }
    scores["overall"] = round(sum(scores.values()) / len(scores), 3)
    return scores


# =============================================================================
# LLM caller — with overhead tracking (not used in scoring)
# =============================================================================

def call_llm(alert: dict, model_name: str, use_rag: bool = False) -> tuple:
    """
    Call LLM. Returns (response, elapsed, overhead_dict).
    overhead_dict contains RAM/token data — NOT used in scoring.
    """
    import requests

    attack_type  = alert.get("attack_type", "unknown")
    tool_results = {
        "attack_details": get_attack_details(attack_type),
        "mitre":          lookup_mitre(attack_type),
        "severity":       estimate_severity(
                              alert.get("anomaly_score", 0),
                              alert.get("lgbm_confidence", 0),
                              alert.get("anomaly_threshold", 1),
                              attack_type
                          ),
        "mitigation":     suggest_mitigation(attack_type),
    }

    rag_block = ""
    if use_rag:
        try:
            from rag.retriever import retrieve
            rag_block = retrieve(
                f"{attack_type} IoT attack mitigation", top_k=1
            )[:100]
        except Exception:
            rag_block = ""

    mitre      = tool_results["mitre"].get("technique", {})
    severity   = tool_results["severity"]
    mitigation = tool_results["mitigation"].get("mitigation_steps", [])
    details    = tool_results["attack_details"].get("details", "Unknown.")
    mit_text   = " | ".join(mitigation[:3])

    ctx_size = 1024 if use_rag else 512

    user = f"""ALERT: {alert['attack_type']} | {alert['severity']}
LightGBM: {alert['lgbm_label']} ({alert['lgbm_confidence']*100:.0f}%)
Anomaly: {alert['is_anomaly']} (score:{alert['anomaly_score']})
{rag_block}
INTELLIGENCE:
Details: {details[:150]}
MITRE: {mitre.get('id','N/A')} - {mitre.get('name','N/A')} ({mitre.get('tactic','N/A')})
CVSS: {severity.get('cvss_score','N/A')} ({severity.get('cvss_vector','N/A')})
Mitigation: {mit_text[:100]}

Write security report:
1) Attack Summary 2) MITRE Mapping 3) CVEs 4) Severity 5) Mitigation Steps"""

    messages = [
        {"role": "system", "content": "You are an IoT security analyst. Write a structured security report. Be concise."},
        {"role": "user",   "content": f"/no_think\n{user}"},
    ]
    payload = {
        "model":    model_name,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": ctx_size, "temperature": 0.3, "num_predict": 256}
    }

    # RAM measurement (for overhead table only — not in scoring)
    process    = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 ** 2)

    start = time.time()
    try:
        resp    = requests.post(f"{config.LLM_HOST}/api/chat",
                                 json=payload, timeout=config.LLM_TIMEOUT)
        elapsed = round(time.time() - start, 2)
        raw     = resp.json()
        answer  = raw.get("message", {}).get("content", "")

        import re
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        if not answer:
            answer = raw.get("message", {}).get("content", "")

        prompt_tokens   = raw.get("prompt_eval_count", 0)
        response_tokens = raw.get("eval_count", 0)
        total_tokens    = prompt_tokens + response_tokens

    except Exception as e:
        elapsed         = round(time.time() - start, 2)
        answer          = f"Error: {str(e)}"
        prompt_tokens   = 0
        response_tokens = 0
        total_tokens    = 0

    ram_after = process.memory_info().rss / (1024 ** 2)
    ram_used  = round(ram_after - ram_before, 2)

    overhead = {
        "prompt_tokens":   prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens":    total_tokens,
        "ram_used_mb":     ram_used,
    }
    return answer, elapsed, overhead


# =============================================================================
# Main evaluation
# =============================================================================

def evaluate(n_samples: int = 10,
             base_model: str = "qwen3:4b",
             ft_model:   str = "iot-ids-llm"):

    print("\n" + "="*65)
    print("EVALUATOR — Base vs Fine-tuned vs Fine-tuned + RAG")
    print("="*65)

    # Load IDS models
    print("\n[Eval] Loading IDS models...")
    with open(os.path.join(config.MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(config.MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    lgbm_model             = lgbm_load()
    ae_model, ae_threshold = ae_load()
    _, X_test, _, y_test, _, _ = preprocess()

    # Sample mixed flows
    benign_idx     = le.transform(["Benign"])[0] if "Benign" in le.classes_ else -1
    benign_indices = np.where(y_test == benign_idx)[0] if benign_idx >= 0 else np.array([])
    attack_indices = np.where(y_test != benign_idx)[0]

    sampled = []
    if n_samples == 1:
        sampled = np.random.choice(attack_indices, 1, replace=False).tolist()
    else:
        n_each = max(1, n_samples // 2)
        if len(benign_indices) > 0:
            sampled.extend(np.random.choice(benign_indices,
                            min(n_each, len(benign_indices)), replace=False).tolist())
        sampled.extend(np.random.choice(attack_indices,
                        min(n_each, len(attack_indices)), replace=False).tolist())

    indices = np.random.permutation(sampled)

    configs = [
        {"name": "Base Model",       "model": base_model, "rag": False},
        {"name": "Fine-tuned",       "model": ft_model,   "rag": False},
        {"name": "Fine-tuned + RAG", "model": ft_model,   "rag": True},
    ]

    results = {c["name"]: {
        "scores":    [],
        "times":     [],
        "overhead":  [],
        "responses": []
    } for c in configs}

    alerts_evaluated = 0

    for i, idx in enumerate(indices):
        features    = X_test[idx]
        lgbm_result = lgbm_predict(lgbm_model, features, le)
        ae_result   = ae_predict(ae_model, ae_threshold, features)
        alert       = build_alert(features, feature_names, lgbm_result, ae_result)
        if alert is None:
            continue

        alerts_evaluated += 1
        print(f"\n[Eval] Alert {alerts_evaluated}: "
              f"{alert['attack_type']} | {alert['severity']}")

        for cfg in configs:
            print(f"  → {cfg['name']}...")
            response, elapsed, overhead = call_llm(alert, cfg["model"], cfg["rag"])
            scores = score_response(response, alert)

            results[cfg["name"]]["scores"].append(scores)
            results[cfg["name"]]["times"].append(elapsed)
            results[cfg["name"]]["overhead"].append(overhead)
            results[cfg["name"]]["responses"].append({
                "alert":    alert["attack_type"],
                "response": response[:500],
                "scores":   scores,
                "time":     elapsed,
                "overhead": overhead,
            })
            print(f"     Score: {scores['overall']*100:.0f}% | "
                  f"Time: {elapsed}s | "
                  f"Tokens: {overhead['total_tokens']} | "
                  f"RAM: {overhead['ram_used_mb']}MB")

    # ── Compute averages ──────────────────────────────────────────────────────
    summary = {}
    for cfg in configs:
        name   = cfg["name"]
        scores = results[name]["scores"]
        times  = results[name]["times"]
        ohs    = results[name]["overhead"]
        if not scores:
            continue
        avg_scores = {}
        for metric in scores[0].keys():
            avg_scores[metric] = round(
                sum(s[metric] for s in scores) / len(scores), 3)
        avg_scores["avg_response_time_s"] = round(sum(times) / len(times), 2)
        # Overhead averages — stored separately for printing only
        avg_scores["avg_prompt_tokens"]   = round(sum(o["prompt_tokens"]   for o in ohs) / len(ohs), 1)
        avg_scores["avg_response_tokens"] = round(sum(o["response_tokens"] for o in ohs) / len(ohs), 1)
        avg_scores["avg_total_tokens"]    = round(sum(o["total_tokens"]    for o in ohs) / len(ohs), 1)
        avg_scores["avg_ram_used_mb"]     = round(sum(o["ram_used_mb"]     for o in ohs) / len(ohs), 2)
        summary[name] = avg_scores

    col_w = 22

    # ── Quality scores table (same as original) ───────────────────────────────
    print("\n\n" + "="*65)
    print("EVALUATION RESULTS — Quality Scores")
    print("="*65)

    quality_metrics = [
        ("mitre_accuracy",     "MITRE Accuracy"),
        ("severity_correct",   "Severity Correct"),
        ("completeness",       "Completeness"),
        ("mitigation_quality", "Mitigation Quality"),
        ("cve_coverage",       "CVE Coverage"),
        ("response_length",    "Response Detail"),
        ("overall",            "Overall Score"),
        ("avg_response_time_s","Avg Response Time (s)"),
    ]

    header = f"{'Metric':<25}" + "".join(f"{c['name']:<{col_w}}" for c in configs)
    print(header)
    print("-" * (25 + col_w * len(configs)))
    for key, label in quality_metrics:
        row = f"{label:<25}"
        for cfg in configs:
            val = summary.get(cfg["name"], {}).get(key, 0)
            if key == "avg_response_time_s":
                row += f"{val:<{col_w}.1f}"
            else:
                row += f"{val*100:<{col_w}.1f}%"
        print(row)
    print("="*65)

    # ── Computational overhead table (printed only — not in scoring) ──────────
    print("\n" + "="*65)
    print("COMPUTATIONAL OVERHEAD  (informational only — not in scoring)")
    print("="*65)

    overhead_metrics = [
        ("avg_prompt_tokens",   "Avg Prompt Tokens"),
        ("avg_response_tokens", "Avg Response Tokens"),
        ("avg_total_tokens",    "Avg Total Tokens"),
        ("avg_ram_used_mb",     "Avg RAM Used (MB)"),
    ]

    header = f"{'Metric':<25}" + "".join(f"{c['name']:<{col_w}}" for c in configs)
    print(header)
    print("-" * (25 + col_w * len(configs)))
    for key, label in overhead_metrics:
        row = f"{label:<25}"
        for cfg in configs:
            val = summary.get(cfg["name"], {}).get(key, 0)
            row += f"{val:<{col_w}.2f}"
        print(row)
    print("="*65)

    # ── Model size table (printed only — not in scoring) ──────────────────────
    print("\n" + "="*65)
    print("MODEL SIZE ON DISK  (informational only — not in scoring)")
    print("="*65)
    print(f"{'Model':<32} {'Format':<14} {'Size':<10} {'Edge Deploy'}")
    print("-"*65)
    print(f"{'qwen3:4b (Base)':<32} {'Q4_K_M GGUF':<14} {'~2.5 GB':<10} Yes (RPi5/Jetson)")
    print(f"{'iot-ids-llm (Fine-tuned)':<32} {'Q4_K_M GGUF':<14} {'~2.5 GB':<10} Yes (RPi5/Jetson)")
    print(f"{'all-MiniLM-L6-v2 (RAG)':<32} {'PyTorch':<14} {'~90 MB':<10} Yes")
    print(f"{'FAISS Index':<32} {'Binary':<14} {'~varies':<10} Yes")
    print("="*65)

    # ── Save reports ──────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts        = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(REPORT_DIR, f"eval_{ts}.json")
    md_path   = os.path.join(REPORT_DIR, f"eval_{ts}.md")

    with open(json_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    # Markdown report
    md  = f"# Evaluation Report\n**Generated:** {ts} UTC\n\n"
    md += "## Quality Scores\n\n"
    md += "| Metric | Base Model | Fine-tuned | Fine-tuned + RAG |\n"
    md += "|--------|-----------|------------|------------------|\n"
    for key, label in quality_metrics:
        row = f"| {label} |"
        for cfg in configs:
            val = summary.get(cfg["name"], {}).get(key, 0)
            if key == "avg_response_time_s":
                row += f" {val:.1f}s |"
            else:
                row += f" {val*100:.1f}% |"
        md += row + "\n"

    md += "\n## Computational Overhead\n\n"
    md += "| Metric | Base Model | Fine-tuned | Fine-tuned + RAG |\n"
    md += "|--------|-----------|------------|------------------|\n"
    for key, label in overhead_metrics:
        row = f"| {label} |"
        for cfg in configs:
            val = summary.get(cfg["name"], {}).get(key, 0)
            row += f" {val:.2f} |"
        md += row + "\n"

    md += "\n## Model Size on Disk\n\n"
    md += "| Model | Format | Size | Edge Deployable |\n"
    md += "|-------|--------|------|-----------------|\n"
    md += "| qwen3:4b (Base) | Q4_K_M GGUF | ~2.5 GB | Yes |\n"
    md += "| iot-ids-llm (Fine-tuned) | Q4_K_M GGUF | ~2.5 GB | Yes |\n"
    md += "| all-MiniLM-L6-v2 (RAG embedder) | PyTorch | ~90 MB | Yes |\n"
    md += "| FAISS Index | Binary | ~varies | Yes |\n"

    md += "\n## Per-Alert Details\n\n"
    for cfg in configs:
        md += f"### {cfg['name']}\n\n"
        for item in results[cfg["name"]]["responses"]:
            md += f"**Alert:** {item['alert']} | "
            md += f"**Score:** {item['scores']['overall']*100:.0f}% | "
            md += f"**Time:** {item['time']}s | "
            md += f"**Tokens:** {item['overhead']['total_tokens']}\n\n"
            md += f"{item['response'][:300]}...\n\n---\n\n"

    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n[Eval] Reports saved:")
    print(f"  JSON → {json_path}")
    print(f"  MD   → {md_path}")
    print(f"  Alerts evaluated: {alerts_evaluated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT IDS Evaluator")
    parser.add_argument("--samples",    type=int, default=10)
    parser.add_argument("--base-model", type=str, default="qwen3:4b")
    parser.add_argument("--ft-model",   type=str, default="iot-ids-llm")
    args = parser.parse_args()
    evaluate(n_samples=args.samples,
             base_model=args.base_model,
             ft_model=args.ft_model)