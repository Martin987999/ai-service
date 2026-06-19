"""Build the bilingual KB corpus + QA evaluation set.

公开测试集:RGB (Retrieval-Augmented Generation Benchmark, 中英双语, Chen et al. 2024)。
  --source rgb   : 尝试用 HuggingFace datasets 拉取 RGB(需联网),映射为内部 KB 语料 + QA 评估集。
  --source builtin (默认): 使用内置中英双语样例语料(无需联网),保证 `一键评估` 可复现。

输出:
  data/corpus/*.jsonl       检索语料(每行 {id,title,text,lang})
  eval/datasets/qa_eval.jsonl  评估集(每行 {id,question,answer/ground_truths,lang,out_of_scope})
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Built-in bilingual sample corpus (employee handbook / compliance / tech spec /
# architecture). Small but real-shaped; lets the whole pipeline + eval run offline.
# ---------------------------------------------------------------------------
BUILTIN_DOCS = [
    {"id": "handbook-leave-zh", "title": "员工手册-年假", "lang": "zh",
     "text": "年假政策:入职满一年的员工每年享有 10 天带薪年假;满五年增加至 15 天;满十年为 20 天。"
             "年假需提前三个工作日在 OA 系统申请,经直属主管审批后生效。当年未休年假最多可结转 5 天至次年第一季度,逾期作废。"},
    {"id": "handbook-remote-zh", "title": "员工手册-远程办公", "lang": "zh",
     "text": "远程办公:员工每周最多可申请 2 天远程办公,需提前在 OA 登记。涉及核心机密数据的岗位不适用远程办公。"
             "远程办公期间须使用公司 VPN 接入内网,且保持即时通讯在线。"},
    {"id": "compliance-pii-zh", "title": "合规指南-个人信息保护", "lang": "zh",
     "text": "个人信息处理:收集用户个人信息须遵循最小必要原则并取得授权同意。敏感个人信息(身份证号、银行卡号、生物识别)"
             "须加密存储,访问需走审批并留痕。数据出境须通过安全评估。违规处理个人信息将依《个人信息保护法》追责。"},
    {"id": "techspec-api-auth-zh", "title": "技术规格-接口鉴权", "lang": "zh",
     "text": "接口鉴权:所有对外 API 使用 OAuth 2.0 客户端凭证模式签发的访问令牌(access token),有效期 2 小时。"
             "令牌通过 Authorization: Bearer 头传递。服务端对每个请求校验令牌签名与作用域(scope),超限返回 429。"},
    {"id": "arch-rag-zh", "title": "架构文档-RAG服务", "lang": "zh",
     "text": "RAG 服务架构:由检索层(向量库 FAISS + 词法 BM25,RRF 融合)、重排层(rerank-2)、生成层(Claude)、"
             "缓存层(精确+语义)与可观测层(结构化日志)组成。单实例支持至少 5 并发,90% 请求 10 秒内完成。"},
    {"id": "handbook-leave-en", "title": "Handbook - Annual Leave", "lang": "en",
     "text": "Annual leave policy: employees with one full year of service receive 10 paid leave days per year; "
             "five years raises it to 15 days; ten years to 20 days. Leave must be requested in the OA system at least "
             "three working days in advance and approved by the direct manager. Up to 5 unused days may carry over to "
             "Q1 of the next year; the rest expire."},
    {"id": "handbook-remote-en", "title": "Handbook - Remote Work", "lang": "en",
     "text": "Remote work: employees may request up to 2 remote days per week, registered in advance in OA. Roles handling "
             "core confidential data are not eligible. While remote, employees must connect to the intranet via the company "
             "VPN and stay reachable on instant messaging."},
    {"id": "compliance-pii-en", "title": "Compliance - Personal Data", "lang": "en",
     "text": "Personal data handling: collection must follow the minimal-necessary principle and obtain consent. Sensitive "
             "personal data (ID numbers, bank cards, biometrics) must be stored encrypted, with approval and audit logging "
             "for access. Cross-border transfer requires a security assessment. Violations are pursued under PIPL."},
    {"id": "techspec-api-auth-en", "title": "Tech Spec - API Authentication", "lang": "en",
     "text": "API authentication: all external APIs use OAuth 2.0 client-credentials access tokens valid for 2 hours, "
             "passed via the Authorization: Bearer header. The server validates token signature and scope on every request "
             "and returns 429 when rate limits are exceeded."},
    {"id": "arch-rag-en", "title": "Architecture - RAG Service", "lang": "en",
     "text": "RAG service architecture: a retrieval layer (FAISS vectors + BM25 lexical, fused with RRF), a rerank layer "
             "(rerank-2), a generation layer (Claude), a cache layer (exact + semantic), and an observability layer "
             "(structured logs). A single instance supports at least 5 concurrent requests with 90% under 10 seconds."},
]

# Eval set: in-scope (with ground-truth) + out-of-scope (expected refusal).
BUILTIN_EVAL = [
    {"id": "q1", "lang": "zh", "question": "入职满五年每年有多少天年假?",
     "ground_truths": ["15 天", "15天"], "out_of_scope": False},
    {"id": "q2", "lang": "zh", "question": "每周最多可以申请几天远程办公?",
     "ground_truths": ["2 天", "两天", "2天"], "out_of_scope": False},
    {"id": "q3", "lang": "zh", "question": "对外 API 的访问令牌有效期多久?",
     "ground_truths": ["2 小时", "两小时", "2小时"], "out_of_scope": False},
    {"id": "q4", "lang": "zh", "question": "敏感个人信息需要怎样存储?",
     "ground_truths": ["加密存储", "加密"], "out_of_scope": False},
    {"id": "q5", "lang": "zh", "question": "未休的年假最多可以结转几天?",
     "ground_truths": ["5 天", "5天"], "out_of_scope": False},
    {"id": "q6", "lang": "en", "question": "How many annual leave days after ten years of service?",
     "ground_truths": ["20 days", "20"], "out_of_scope": False},
    {"id": "q7", "lang": "en", "question": "How is the API access token transmitted?",
     "ground_truths": ["Authorization: Bearer", "Bearer header", "Bearer"], "out_of_scope": False},
    {"id": "q8", "lang": "en", "question": "What fusion method does the retrieval layer use?",
     "ground_truths": ["RRF", "Reciprocal Rank Fusion"], "out_of_scope": False},
    # out-of-scope → expected refusal
    {"id": "q9", "lang": "zh", "question": "公司今年的股票分红是多少?",
     "ground_truths": [], "out_of_scope": True},
    {"id": "q10", "lang": "en", "question": "What is the CEO's home address?",
     "ground_truths": [], "out_of_scope": True},
]


def write_builtin(corpus_dir: Path, eval_path: Path) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_dir / "kb_builtin.jsonl", "w", encoding="utf-8") as f:
        for d in BUILTIN_DOCS:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for q in BUILTIN_EVAL:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"[make_corpus] wrote {len(BUILTIN_DOCS)} docs -> {corpus_dir}")
    print(f"[make_corpus] wrote {len(BUILTIN_EVAL)} eval items -> {eval_path}")


def write_rgb(corpus_dir: Path, eval_path: Path, limit: int = 200) -> None:
    """Fetch RGB (bilingual RAG benchmark) and map to KB corpus + QA eval set."""
    from datasets import load_dataset

    corpus_dir.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_eval = 0
    with open(corpus_dir / "kb_rgb.jsonl", "w", encoding="utf-8") as cf, \
         open(eval_path, "w", encoding="utf-8") as ef:
        for lang, name in (("en", "en"), ("zh", "zh")):
            ds = load_dataset("chenxwh/RGB", name, split="test")  # positive passages + QA
            for i, row in enumerate(ds):
                if i >= limit:
                    break
                # RGB rows carry a question, an answer, and supporting passages.
                passages = row.get("positive") or row.get("context") or []
                for j, p in enumerate(passages if isinstance(passages, list) else [passages]):
                    cf.write(json.dumps({"id": f"rgb-{lang}-{i}-{j}", "title": "",
                                         "text": p, "lang": lang}, ensure_ascii=False) + "\n")
                    n_docs += 1
                ans = row.get("answer")
                gts = ans if isinstance(ans, list) else [ans]
                ef.write(json.dumps({"id": f"rgb-{lang}-{i}", "lang": lang,
                                     "question": row.get("query") or row.get("question"),
                                     "ground_truths": [g for g in gts if g],
                                     "out_of_scope": False}, ensure_ascii=False) + "\n")
                n_eval += 1
    print(f"[make_corpus] RGB: wrote {n_docs} passages, {n_eval} eval items")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["builtin", "rgb"], default="builtin")
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--eval-path", default="eval/datasets/qa_eval.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    corpus_dir, eval_path = Path(args.corpus_dir), Path(args.eval_path)
    if args.source == "rgb":
        try:
            write_rgb(corpus_dir, eval_path, args.limit)
            return
        except Exception as e:
            print(f"[make_corpus] RGB download failed ({e}); falling back to builtin.")
    write_builtin(corpus_dir, eval_path)


if __name__ == "__main__":
    main()
