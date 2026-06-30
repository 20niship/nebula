#!/usr/bin/env python3
"""Mac (MoltenVK) 向け PNanoVDB.h の double → float 自動変換スクリプト。

修正内容:
  1. GLSL モードの double 型関数を float スタブに置換
  2. 全 double → float
  3. PNANOVDB_STATIC_CONST uint 配列を GLSL 配列コンストラクタ構文に変換
     (C 式の部分初期化と int→uint 暗黙変換が GLSL で NG なため)
  4. pnanovdb_grid_type_constants_t 構造体配列を GLSL コンストラクタ構文に変換
"""
import re
import sys


def add_u_suffix(vals_str: str) -> str:
    """整数リテラルに u サフィックスを付ける。すでに u がある場合はスキップ。"""
    tokens = re.split(r'(\s*,\s*|\s+)', vals_str)
    out = []
    for tok in tokens:
        num = re.fullmatch(r'(\s*)([-+]?\d+)(\s*)', tok)
        if num:
            out.append(f"{num.group(1)}{num.group(2)}u{num.group(3)}")
        else:
            out.append(tok)
    return ''.join(out)


def parse_brace_block(text: str, start: int):
    """start の位置にある '{...}' ブロックの内容と終端位置を返す。"""
    assert text[start] == '{', f"Expected '{{' at {start}, got {text[start]!r}"
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i
        i += 1
    raise ValueError("Unmatched '{'")


def fix_uint_array(match: re.Match) -> str:
    """PNANOVDB_STATIC_CONST uint arr[N] = { ... }; を GLSL 配列コンストラクタに変換"""
    cap_name = 'PNANOVDB_GRID_TYPE_CAP'  # = 32
    arr_name = match.group('name')
    inner = match.group('inner')
    vals = [v.strip() for v in inner.split(',') if v.strip()]
    # 32 要素に足りない分は 0u でパディング
    n = 32
    while len(vals) < n:
        vals.append('0')
    vals_u = ', '.join(v if v.endswith('u') else v + 'u' for v in vals[:n])
    return (
        f"const uint {arr_name}[{cap_name}] = uint[{cap_name}](\n"
        f"    {vals_u}\n"
        f");"
    )


def fix_struct_array(text: str) -> str:
    """pnanovdb_grid_type_constants_t 配列を GLSL コンストラクタ構文に変換"""
    struct_type = 'pnanovdb_grid_type_constants_t'
    cap = 'PNANOVDB_GRID_TYPE_CAP'
    marker = f'PNANOVDB_STATIC_CONST {struct_type} pnanovdb_grid_type_constants[{cap}] ='
    idx = text.find(marker)
    if idx == -1:
        return text

    # マーカーの後の '{' を探してブロック全体を取得
    brace_start = text.index('{', idx + len(marker))
    inner, brace_end = parse_brace_block(text, brace_start)

    # 各エントリを分解 (C スタイル: {v1, v2, ...} が28個並んでいる)
    entries = []
    i = 0
    while i < len(inner):
        if inner[i] == '{':
            entry_inner, end = parse_brace_block(inner, i)
            vals = [v.strip() for v in entry_inner.split(',') if v.strip()]
            # 28 フィールドに揃える
            while len(vals) < 28:
                vals.append('0')
            vals_u = ', '.join(v + 'u' if not v.endswith('u') else v for v in vals[:28])
            entries.append(f"    {struct_type}({vals_u})")
            i = end + 1
        else:
            i += 1

    # 32 エントリに足りない分はゼロ構造体でパディング
    zero_fields = ', '.join(['0u'] * 28)
    zero_entry = f"    {struct_type}({zero_fields})"
    while len(entries) < 32:
        entries.append(zero_entry)

    replacement = (
        f"const {struct_type} pnanovdb_grid_type_constants[{cap}] = "
        f"{struct_type}[{cap}](\n"
        + ',\n'.join(entries[:32]) + '\n'
        + ");"
    )

    # brace_end は '}' の位置。直後の ';' も消費してダブルセミコロンを防ぐ
    tail_start = brace_end + 1
    if tail_start < len(text) and text[tail_start] == ';':
        tail_start += 1
    return text[:idx] + replacement + text[tail_start:]


# ── メイン処理 ──────────────────────────────────────────────────────────────

content = open(sys.argv[1]).read()

# 1. GLSL モード内の packDouble2x32 / unpackDouble2x32 を安全な float 実装で置換
content = content.replace(
    'double pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return packDouble2x32(uvec2(v.x, v.y)); }',
    'float pnanovdb_uint64_as_double(pnanovdb_uint64_t v) { return uintBitsToFloat(v.x); }'
)
content = content.replace(
    'pnanovdb_uint64_t pnanovdb_double_as_uint64(double v) { return unpackDouble2x32(v); }',
    'pnanovdb_uint64_t pnanovdb_double_as_uint64(float v) { return uvec2(floatBitsToUint(v), 0u); }'
)

# 2. 残りの double → float
content = re.sub(r'\bdouble\b', 'float', content)

# 3. PNANOVDB_STATIC_CONST uint 1行配列を GLSL コンストラクタ構文に変換
#    例: PNANOVDB_STATIC_CONST pnanovdb_uint32_t foo[CAP] = { 0, 32, ... };
uint_pat = re.compile(
    r'PNANOVDB_STATIC_CONST\s+pnanovdb_uint32_t\s+(?P<name>\w+)\[PNANOVDB_GRID_TYPE_CAP\]\s*=\s*\{(?P<inner>[^}]*)\}\s*;',
    re.DOTALL
)
content = uint_pat.sub(fix_uint_array, content)

# 4. pnanovdb_grid_type_constants_t 配列を GLSL コンストラクタ構文に変換
content = fix_struct_array(content)

open(sys.argv[2], 'w').write(content)
print(f"Patched {sys.argv[1]} → {sys.argv[2]}")
