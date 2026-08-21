"""Audit whether the project datasets expose fraud-semantic factors.

The audit uses conservative lexical indicators as an observable-signal proxy.
It does not treat keyword matches as ground-truth factor annotations.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_PATH = PROJECT_ROOT / "outputs" / "semantic_factor_audit.md"
JSON_PATH = PROJECT_ROOT / "outputs" / "semantic_factor_audit.json"

TELECOM_LABELS = {
    "label00-last.csv": "正常文本",
    "label01-last.csv": "冒充公检法及政府机关类",
    "label02-last.csv": "贷款、代办信用卡类",
    "label03-last.csv": "冒充电商物流客服类",
    "label04-last.csv": "冒充领导、熟人类",
}

FACTOR_TERMS = {
    "identity_scenario": [
        "公安",
        "警察",
        "警官",
        "检察院",
        "法院",
        "政府",
        "客服",
        "平台",
        "淘宝",
        "天猫",
        "京东",
        "支付宝",
        "微信",
        "抖音",
        "快手",
        "快递",
        "物流",
        "银行",
        "贷款",
        "借款",
        "征信",
        "信用卡",
        "领导",
        "老板",
        "熟人",
        "朋友",
        "亲戚",
        "同学",
        "老师",
        "军人",
        "军官",
        "部队",
        "投资",
        "理财",
        "股票",
        "基金",
        "虚拟币",
        "恋爱",
        "交友",
        "婚恋",
        "购物",
        "订单",
        "商家",
        "疫情",
        "防控",
        "招聘",
        "兼职",
    ],
    "manipulation_tactic": [
        "立即",
        "马上",
        "尽快",
        "紧急",
        "限时",
        "逾期",
        "最后期限",
        "过期",
        "否则",
        "涉嫌",
        "犯罪",
        "案件",
        "通缉",
        "冻结",
        "调查",
        "配合",
        "违规",
        "处罚",
        "法律责任",
        "安全账户",
        "保密",
        "中奖",
        "返利",
        "高额",
        "高收益",
        "稳赚",
        "赚钱",
        "盈利",
        "优惠",
        "低息",
        "免息",
        "退款",
        "理赔",
        "赔偿",
        "奖励",
        "免单",
        "免费",
        "验证身份",
        "恢复信用",
        "消除征信",
        "帮忙",
        "急用",
        "周转",
    ],
    "requested_action": [
        "转账",
        "汇款",
        "付款",
        "支付",
        "充值",
        "垫付",
        "保证金",
        "手续费",
        "解冻金",
        "认证金",
        "验证码",
        "身份证",
        "银行卡",
        "密码",
        "卡号",
        "点击",
        "链接",
        "扫码",
        "扫描二维码",
        "二维码",
        "添加微信",
        "加微信",
        "添加好友",
        "联系客服",
        "下载",
        "安装",
        "登录",
        "注册",
        "拨打",
        "打开",
        "提交",
        "填写",
        "提供信息",
        "提供资料",
    ],
}

FACTOR_NAMES = {
    "identity_scenario": "身份/情境",
    "manipulation_tactic": "操纵策略",
    "requested_action": "行为请求",
}


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {path}")


def load_telecom5() -> list[dict[str, str]]:
    data_dir = DATA_ROOT / "Telecom_Fraud_Texts_5"
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for filename, label in TELECOM_LABELS.items():
        for row in read_csv_rows(data_dir / filename):
            text = normalize_text(row.get("content", ""))
            key = (text, label)
            if text and key not in seen:
                seen.add(key)
                records.append({"text": text, "label": label})
    return records


def load_fgrc_scd() -> list[dict[str, str]]:
    path = DATA_ROOT / "FGRC-SCD" / "sms" / "message" / "finetuning_initial.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        text = normalize_text(row.get("文本", ""))
        label = normalize_text(row.get("风险类别", ""))
        key = (text, label)
        if text and label and key not in seen:
            seen.add(key)
            records.append({"text": text, "label": label})
    return records


def load_spam_message() -> list[dict[str, str]]:
    path = DATA_ROOT / "SpamMessage" / "SpamMessage-master" / "data" / "带标签短信.txt"
    labels = {"0": "正常短信", "1": "垃圾短信"}
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.rstrip("\n")
            if "\t" not in raw:
                continue
            raw_label, text = raw.split("\t", 1)
            label = labels.get(raw_label.strip())
            text = normalize_text(text)
            if not text or label is None:
                continue
            key = (text, label)
            if key not in seen:
                seen.add(key)
                records.append({"text": text, "label": label})
    return records


def compile_patterns() -> dict[str, re.Pattern[str]]:
    return {
        factor: re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
        for factor, terms in FACTOR_TERMS.items()
    }


def matched_terms(text: str, factor: str) -> list[str]:
    lowered = text.lower()
    return [term for term in FACTOR_TERMS[factor] if term.lower() in lowered]


def audit_dataset(
    records: list[dict[str, str]], patterns: dict[str, re.Pattern[str]]
) -> dict[str, object]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)

    class_results: dict[str, object] = {}
    for label, rows in sorted(by_label.items()):
        counts = Counter()
        flagged_rows: list[dict[str, object]] = []
        for row in rows:
            flags = {factor: bool(pattern.search(row["text"])) for factor, pattern in patterns.items()}
            hit_count = sum(flags.values())
            for factor, present in flags.items():
                counts[factor] += int(present)
            counts["at_least_two"] += int(hit_count >= 2)
            counts["all_three"] += int(hit_count == 3)
            counts["none"] += int(hit_count == 0)
            flagged_rows.append(
                {
                    "text": row["text"],
                    "flags": flags,
                    "terms": {
                        factor: matched_terms(row["text"], factor)
                        for factor in FACTOR_TERMS
                    },
                }
            )

        rng = random.Random(f"{label}-2026")
        sample_pool = flagged_rows[:]
        rng.shuffle(sample_pool)
        samples = sample_pool[:3]
        total = len(rows)
        class_results[label] = {
            "count": total,
            "coverage": {
                key: {
                    "count": counts[key],
                    "rate": counts[key] / total if total else 0.0,
                }
                for key in (*FACTOR_TERMS.keys(), "at_least_two", "all_three", "none")
            },
            "samples": samples,
        }

    return {
        "sample_count": len(records),
        "class_count": len(class_results),
        "classes": class_results,
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def truncate(text: str, limit: int = 125) -> str:
    text = text.replace("|", "｜")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def iter_report_lines(results: dict[str, object]) -> Iterable[str]:
    yield "# 数据集诈骗语义因子审计报告"
    yield ""
    yield "本报告检查原始文本中是否可观察到“身份/情境、操纵策略、行为请求”三类信号。"
    yield "统计依据是人工构建的保守关键词词表，因此只能作为可观测性诊断，不能当作真实因子标签或最终实验结论。"
    yield ""

    for dataset_name, result in results.items():
        yield f"## {dataset_name}"
        yield ""
        yield f"- 去重样本数：{result['sample_count']}"
        yield f"- 类别数：{result['class_count']}"
        yield ""
        yield "| 类别 | 样本数 | 身份/情境 | 操纵策略 | 行为请求 | 至少两类信号 | 三类信号共现 | 均未命中 |"
        yield "|---|---:|---:|---:|---:|---:|---:|---:|"
        for label, class_result in result["classes"].items():
            coverage = class_result["coverage"]
            yield (
                f"| {label} | {class_result['count']} | "
                f"{pct(coverage['identity_scenario']['rate'])} | "
                f"{pct(coverage['manipulation_tactic']['rate'])} | "
                f"{pct(coverage['requested_action']['rate'])} | "
                f"{pct(coverage['at_least_two']['rate'])} | "
                f"{pct(coverage['all_three']['rate'])} | "
                f"{pct(coverage['none']['rate'])} |"
            )
        yield ""
        yield "### 随机核查样本"
        yield ""
        for label, class_result in result["classes"].items():
            yield f"**{label}**"
            yield ""
            for sample in class_result["samples"]:
                present = [
                    FACTOR_NAMES[factor]
                    for factor, matched in sample["flags"].items()
                    if matched
                ]
                signal_text = "、".join(present) if present else "未命中"
                yield f"- [{signal_text}] {truncate(sample['text'])}"
            yield ""

    yield "## 使用边界"
    yield ""
    yield "1. 因子原型只能在文本中确实存在相应线索的数据集上使用。"
    yield "2. 关键词命中不等于语义标签；正式模型应使用弱监督初始化，并通过训练样本中心和可学习残差进行修正。"
    yield "3. 正常类和垃圾短信也可能包含“限时、点击、优惠”等词，因此必须保留反证据或竞争原型，不能依靠单个触发词分类。"
    yield "4. SpamMessage 的“垃圾短信”并不等同于“电信诈骗”，不应为其强行构造诈骗类型的三因子监督。"


def main() -> None:
    patterns = compile_patterns()
    datasets = {
        "Telecom_Fraud_Texts_5": load_telecom5(),
        "FGRC-SCD": load_fgrc_scd(),
        "SpamMessage": load_spam_message(),
    }
    results = {
        dataset_name: audit_dataset(records, patterns)
        for dataset_name, records in datasets.items()
    }
    JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT_PATH.write_text("\n".join(iter_report_lines(results)) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
