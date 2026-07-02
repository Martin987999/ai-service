# -*- coding: utf-8 -*-
"""Generate a large synthetic bilingual KB corpus for scale testing.

合成大规模双语语料,验证索引/检索可扩展性(默认写到 data/corpus_large,不覆盖样例语料)。
每个主题下生成多篇相关但各异的文档(带不同的数字/部门/系统名),这样:
  ① 语料规模可大(几百~上千篇 → 切块后上千~数千块);
  ② 同主题下有多个相关块,context precision / recall 才有讨论空间(不像 10 块玩具语料)。

用法:
  python -m scripts.gen_large_corpus --n 800 --out data/corpus_large
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (theme_key, zh_title, en_title, zh_template, en_template) — {var} 会被随机实体替换
THEMES = [
    ("leave", "员工手册-年假", "Handbook - Annual Leave",
     "{dept}部门年假政策:入职满{y1}年员工每年享有{d1}天带薪年假,满{y2}年增加至{d2}天。"
     "年假需提前{adv}个工作日在{sys}系统申请,经{role}审批后生效。当年未休年假最多结转{carry}天至次年。",
     "{dept} annual leave: employees with {y1} years get {d1} paid days, rising to {d2} after {y2} years. "
     "Requests go through the {sys} system at least {adv} working days ahead, approved by the {role}. "
     "Up to {carry} unused days carry over to next year."),
    ("remote", "员工手册-远程办公", "Handbook - Remote Work",
     "{dept}远程办公:员工每周最多申请{d1}天远程,需提前在{sys}登记。涉及{data}数据的岗位不适用。"
     "远程期间须使用公司VPN接入内网并保持{tool}在线。",
     "{dept} remote work: up to {d1} remote days per week, registered in {sys}. Roles handling {data} data are "
     "excluded. Employees must use the company VPN and stay reachable on {tool}."),
    ("compliance", "合规指南-数据保护", "Compliance - Data Protection",
     "{dept}数据合规:收集{data}信息须遵循最小必要原则并取得授权。敏感信息须加密存储,访问经{role}审批并留痕。"
     "数据出境须通过安全评估,违规依{law}追责。保留期限为{carry}个月。",
     "{dept} data compliance: collecting {data} data follows the minimal-necessary principle with consent. "
     "Sensitive data is stored encrypted; access is approved by the {role} and logged. Cross-border transfer "
     "needs a security assessment; violations are pursued under {law}. Retention is {carry} months."),
    ("api", "技术规格-接口鉴权", "Tech Spec - API Authentication",
     "{sys}接口鉴权:对外API使用OAuth 2.0签发的访问令牌,有效期{d1}小时,通过Authorization: Bearer头传递。"
     "服务端校验令牌签名与作用域,超过{rate}次/分钟返回429。{role}负责密钥轮换,周期{carry}天。",
     "{sys} API auth: external APIs use OAuth 2.0 access tokens valid for {d1} hours via the Authorization: Bearer "
     "header. The server validates signature and scope; over {rate} req/min returns 429. The {role} rotates keys "
     "every {carry} days."),
    ("arch", "架构文档-服务设计", "Architecture - Service Design",
     "{sys}服务架构:由检索层、{tool}重排层、生成层与缓存层组成。单实例支持至少{d1}并发,"
     "{pct}%请求在{d2}秒内完成。降级策略由{role}维护,故障切换{adv}秒内完成。",
     "{sys} architecture: retrieval layer, {tool} rerank layer, generation layer, cache layer. A single instance "
     "supports at least {d1} concurrent requests with {pct}% under {d2} seconds. Failover completes within {adv}s, "
     "maintained by the {role}."),
    ("security", "安全规范-访问控制", "Security - Access Control",
     "{sys}访问控制:采用RBAC,权限按{role}最小授予。特权操作须二次验证(MFA),会话超时{d1}分钟。"
     "审计日志保留{carry}个月,异常登录由{dept}团队{adv}分钟内响应。",
     "{sys} access control: RBAC with least-privilege per {role}. Privileged actions require MFA; sessions time "
     "out after {d1} minutes. Audit logs are kept {carry} months; the {dept} team responds to anomalous logins "
     "within {adv} minutes."),
    ("itsupport", "IT支持-设备申领", "IT Support - Equipment",
     "{dept}设备申领:新员工可申领笔记本与{tool},通过{sys}提交工单,{role}审批后{adv}个工作日内发放。"
     "设备报修SLA为{d1}小时,借用期最长{carry}天。",
     "{dept} equipment: new hires request a laptop and {tool} via a {sys} ticket, approved by the {role} and "
     "delivered within {adv} working days. Repair SLA is {d1} hours; loans last up to {carry} days."),
    ("finance", "财务制度-报销", "Finance - Reimbursement",
     "{dept}报销制度:差旅费须在{adv}个工作日内通过{sys}提交,附发票,经{role}审批。单笔超过{d1}元需额外审批。"
     "报销周期{carry}天,超期{d2}个月未提交视为放弃。",
     "{dept} reimbursement: travel expenses are submitted via {sys} within {adv} working days with receipts, "
     "approved by the {role}. Amounts over {d1} need extra approval. Cycle is {carry} days; unsubmitted after "
     "{d2} months is forfeited."),
]

DEPTS = ["技术", "人力", "财务", "法务", "运营", "安全", "Engineering", "HR", "Finance", "Legal"]
ROLES = ["直属主管", "部门负责人", "合规官", "安全负责人", "team lead", "manager", "compliance officer"]
SYS = ["OA", "Workday", "Jira", "ServiceNow", "内部门户", "SSO平台"]
TOOLS = ["即时通讯", "显示器", "rerank-2", "Slack", "VPN客户端", "MFA令牌"]
DATA = ["个人", "客户", "财务", "生物识别", "personal", "customer", "PII"]
LAWS = ["《个人信息保护法》", "《数据安全法》", "PIPL", "GDPR"]


def gen(n: int, out_dir: str, seed: int = 42) -> None:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kb_large.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            theme = THEMES[i % len(THEMES)]
            key, zt, et, ztpl, etpl = theme
            lang = "zh" if i % 2 == 0 else "en"
            vars = {
                "dept": rng.choice(DEPTS), "role": rng.choice(ROLES), "sys": rng.choice(SYS),
                "tool": rng.choice(TOOLS), "data": rng.choice(DATA), "law": rng.choice(LAWS),
                "y1": rng.choice([1, 2, 3]), "y2": rng.choice([5, 8, 10]),
                "d1": rng.choice([2, 5, 10, 15, 20]), "d2": rng.choice([10, 15, 20, 30]),
                "adv": rng.choice([1, 2, 3, 5]), "carry": rng.choice([3, 5, 6, 12, 30, 90]),
                "rate": rng.choice([60, 100, 300]), "pct": rng.choice([90, 95, 99]),
            }
            tpl = ztpl if lang == "zh" else etpl
            title = zt if lang == "zh" else et
            text = (title + ":\n") + tpl.format(**vars)
            f.write(json.dumps({"id": f"{key}-{lang}-{i}", "title": title, "text": text,
                                "lang": lang}, ensure_ascii=False) + "\n")
    print(f"[gen_large_corpus] wrote {n} docs -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="data/corpus_large")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    gen(args.n, args.out, args.seed)


if __name__ == "__main__":
    main()
