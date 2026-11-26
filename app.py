import os
# 행렬 계산이 CPU 코어를 전부 차지하지 않도록 계산 스레드를 1개로 제한 (numpy를 불러오기 전에 설정해야 함)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import json
import time
import threading
import random
import html
from concurrent.futures import ThreadPoolExecutor
import re
import streamlit.components.v1 as components
from pathlib import Path
from urllib.parse import urlencode

st.set_page_config(page_title="코스피·코스닥 TOP80 차트", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding:0.8rem 0.4rem 110px 0.4rem;max-width:100%;}  /* 아래 110px 여백: GitHub 화면(index.html)이 아래 72px를 가려도 마지막 줄까지 보이게 */
header,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,
[data-testid="manage-app-button"],[class*="_profileContainer_"],[class*="viewerBadge"],[class*="_terminalButton_"],[class*="_gzau3_"]{display:none !important;}
</style>""", unsafe_allow_html=True)

# ───────────── 차트 계산 설정(고정값) ─────────────
disp, calc, lam, kmul, band_pct = 120, 720, 8e3, 1.6, 0.0


TOP_N = 80   # 상위 몇 등까지 보여줄지 (시가총액 순위 / 거래대금 순위(ETF 포함))


def _num(v):
    """'1,234' / '-1.5' / NaN 등을 숫자(또는 None)로"""
    try:
        f = float(str(v).replace(",", "").replace("%", "").strip())
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _row(code, name, close, chg, ratio, opn, high, low, amount=None, cap=None):
    """리스트 한 줄: [코드, 이름, 현재가, 대비(부호 포함), 등락률, 시가, 고가, 저가, 거래대금(원), 시가총액(원)]"""
    c, r = _num(close), _num(ratio)
    d = _num(chg)
    if d is not None and r is not None:      # 대비의 부호는 등락률 부호로 맞춤(상승 +, 하락 -)
        d = abs(d) * (1 if r > 0 else (-1 if r < 0 else 0))
    return [str(code), str(name), c, d, r, _num(opn), _num(high), _num(low), _num(amount), _num(cap)]


def _won_scale(amount, price, volume) -> float:
    """거래대금 단위를 '원'으로 맞추는 배수. (현재가×거래량)과 비교해 자동으로 판단
    예) 주식 목록은 원 단위 → 1, ETF 목록은 백만원 단위 → 1,000,000"""
    a = pd.to_numeric(pd.Series(amount), errors="coerce").to_numpy(dtype=float)
    pv = (pd.to_numeric(pd.Series(price), errors="coerce") * pd.to_numeric(pd.Series(volume), errors="coerce")).to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(pv) & (a > 0) & (pv > 0)
    if not ok.any():
        return 1.0
    return float(10 ** round(np.log10(np.median(pv[ok] / a[ok]))))


def _etf_rows():
    """국내 ETF 전체 (모두 유가증권시장=코스피 상장). 시가·고가·저가는 목록에 없어서 일봉에서 채움"""
    try:
        e = fdr.StockListing("ETF/KR")
        code, name, price, chg, rate, vol, amt = (e.get("Symbol"), e.get("Name"), e.get("Price"), e.get("Change"),
                                                  e.get("ChangeRate"), e.get("Volume"), e.get("Amount"))
    except Exception:
        import requests
        j = requests.get("https://finance.naver.com/api/sise/etfItemList.nhn",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=8).json()
        e = pd.DataFrame(j["result"]["etfItemList"])
        code, name, price, chg, rate, vol, amt = (e["itemcode"], e["itemname"], e["nowVal"], e["changeVal"],
                                                  e["changeRate"], e["quant"], e["amonut"])
    k = _won_scale(amt, price, vol)
    return [_row(c, n, p, d, r, None, None, None, None if _num(a) is None else _num(a) * k)
            for c, n, p, d, r, a in zip(code, name, price, chg, rate, amt)]


def _top_by_amount(rows, n):
    return sorted(rows, key=lambda r: -(r[8] or 0))[:n]


def _top_by_cap(rows, n):
    return sorted(rows, key=lambda r: -(r[9] or 0))[:n]


KRX_URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"


def _krx(bld: str, date: str, **kw):
    import requests
    r = requests.post(KRX_URL, data={"bld": bld, "locale": "ko_KR", "trdDd": date, "share": "1", "money": "1",
                                     "csvxls_isNo": "false", **kw},
                      headers={"User-Agent": "Mozilla/5.0",
                               "Referer": "http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101"},
                      timeout=10)
    j = r.json()
    return j.get("OutBlock_1") or j.get("output") or []


def _krx_rows(items, scale_from=None):
    """거래소 자료 한 줄씩 → 리스트 한 줄 (거래대금 단위는 현재가×거래량과 비교해 원으로 맞춤)"""
    k = _won_scale([x.get("ACC_TRDVAL") for x in items], [x.get("TDD_CLSPRC") for x in items],
                   [x.get("ACC_TRDVOL") for x in items])
    out = []
    for x in items:
        a = _num(x.get("ACC_TRDVAL"))
        out.append(_row(x.get("ISU_SRT_CD"), x.get("ISU_ABBRV"), x.get("TDD_CLSPRC"), x.get("CMPPREVDD_PRC"),
                        x.get("FLUC_RT"), x.get("TDD_OPNPRC"), x.get("TDD_HGPRC"), x.get("TDD_LWPRC"),
                        None if a is None else a * k, x.get("MKTCAP")))
    return out


def _from_krx(n):
    """거래소 전종목 시세. 장중에는 '오늘' 자료가 비어 있으므로 하루씩 거슬러 가며 가격이 채워진 날짜를 사용"""
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize()
    items, day = [], None
    for back in range(10):
        d = today - pd.Timedelta(days=back)
        if d.weekday() >= 5:                       # 토·일은 건너뜀
            continue
        try:
            got = _krx("dbms/MDC/STAT/standard/MDCSTAT01501", d.strftime("%Y%m%d"), mktId="ALL")
        except Exception:
            got = []
        good = [x for x in got if (_num(x.get("TDD_CLSPRC")) or 0) > 0 and (_num(x.get("MKTCAP")) or 0) > 0]
        if len(good) >= 500:
            items, day = good, d
            break
    if not items:
        raise ValueError("최근 거래소 자료 없음")
    try:                                           # ETF 전종목 (같은 날짜)
        etf_items = [x for x in _krx("dbms/MDC/STAT/standard/MDCSTAT04301", day.strftime("%Y%m%d"))
                     if (_num(x.get("TDD_CLSPRC")) or 0) > 0]
        etfs = _krx_rows(etf_items) if len(etf_items) >= 50 else _etf_rows()
    except Exception:
        try:
            etfs = _etf_rows()
        except Exception:
            etfs = []
    out = {"cap": {}, "amt": {}}
    for m in ("KOSPI", "KOSDAQ"):
        its = [x for x in items if str(x.get("MKT_NM", "")).upper().startswith(m) and
               (m == "KOSDAQ" or str(x.get("MKT_NM", "")).upper() == "KOSPI")]
        rows = _krx_rows(its)
        out["cap"][m] = _top_by_cap(rows, n)
        out["amt"][m] = _top_by_amount(rows + (etfs if m == "KOSPI" else []), n)
    if not out["cap"]["KOSPI"] or not out["cap"]["KOSDAQ"]:
        raise ValueError("빈 목록")
    return out, (None if day == today else day.strftime("%m-%d"))


def _from_fdr(n):
    d = fdr.StockListing("KRX")
    if "Amount" not in d.columns or "Marcap" not in d.columns:
        raise ValueError("거래대금·시가총액 정보 없음")
    # 장 시작 직후 등 거래소 자료가 아직 비어 있으면(종목 이름만 있고 가격·시가총액이 없음) 이 목록은 쓰지 않음
    ok = (pd.to_numeric(d.get("Close"), errors="coerce") > 0) & (pd.to_numeric(d["Marcap"], errors="coerce") > 0)
    if ok.sum() < 500:
        raise ValueError(f"가격·시가총액이 있는 종목이 {int(ok.sum())}개뿐 (자료 준비 전)")
    d = d[ok]
    cc = "Code" if "Code" in d.columns else "Symbol"
    rc = next((c for c in ("ChagesRatio", "ChangesRatio", "ChangeRatio") if c in d.columns), None)
    k = _won_scale(d["Amount"], d.get("Close"), d.get("Volume"))
    try:
        etfs = _etf_rows()
    except Exception:
        etfs = []
    mk = d["Market"].astype(str).str.upper()
    out = {}
    for m in ("KOSPI", "KOSDAQ"):
        s = d[mk.str.startswith("KOSDAQ")] if m == "KOSDAQ" else d[mk == "KOSPI"]
        rows = [_row(r[cc], r["Name"], r.get("Close"), r.get("Changes"), r.get(rc) if rc else None,
                     r.get("Open"), r.get("High"), r.get("Low"),
                     None if _num(r.get("Amount")) is None else _num(r.get("Amount")) * k,
                     r.get("Marcap")) for _, r in s.iterrows()]
        out.setdefault("cap", {})[m] = _top_by_cap(rows, n)       # 시가총액 순위 (주식만)
        if m == "KOSPI":
            try:
                rows += etfs                     # 거래대금 순위는 코스피 목록에 ETF 포함
            except Exception:
                pass
        out.setdefault("amt", {})[m] = _top_by_amount(rows, n)
    if not out["cap"]["KOSPI"] or not out["cap"]["KOSDAQ"]:
        raise ValueError("빈 목록")
    return out


def _from_naver(n):
    """예비용: 네이버 시세(종목 200개씩 받아 거래대금 순으로 정렬) + ETF"""
    import requests
    out = {}
    for m in ("KOSPI", "KOSDAQ"):
        r = requests.get(f"https://m.stock.naver.com/api/stocks/marketValue/{m}",
                         params={"page": 1, "pageSize": 200}, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        st_ = r.json().get("stocks", [])
        amt = [s.get("accumulatedTradingValue") for s in st_]
        k = _won_scale(amt, [s.get("closePrice") for s in st_], [s.get("accumulatedTradingVolume") for s in st_])
        rows = [_row(s.get("itemCode"), s.get("stockName"), s.get("closePrice"),
                     s.get("compareToPreviousClosePrice"), s.get("fluctuationsRatio"),
                     s.get("openPrice"), s.get("highPrice"), s.get("lowPrice"),
                     None if _num(a) is None else _num(a) * k) for s, a in zip(st_, amt)]
        if sum(r[2] is not None for r in rows[:n]) < n * 0.8:
            raise ValueError("네이버 시세도 비어 있음")
        out.setdefault("cap", {})[m] = rows[:n]            # 네이버 목록은 원래 시가총액 순서
        if m == "KOSPI":
            try:
                rows = rows + _etf_rows()
            except Exception:
                pass
        out.setdefault("amt", {})[m] = _top_by_amount(rows, n)
    if not out["cap"]["KOSPI"] or not out["cap"]["KOSDAQ"]:
        raise ValueError("빈 목록")
    return out


def _today_src(f):
    """오늘 자료로 만드는 받기 방법 → (목록, 순위 기준일 None)"""
    def g(n):
        return f(n), None
    g.__name__ = f.__name__
    return g


# ───────────── 한국투자증권 Open API (실시간 시세) ─────────────
# 키는 코드에 넣지 않고 Streamlit의 Secrets(비밀 보관함)에서 읽음:
#   KIS_APP_KEY = "앱키"      KIS_APP_SECRET = "앱시크릿"
KIS_BASE = "https://openapi.koreainvestment.com:9443"     # 실전 서버 (순위 조회는 실전만 가능)


_KIS_KEYS = {}


def _kis_keys():
    """Secrets에서 키 읽기 (한 번 읽으면 기억해 두고, 동시에 보내는 작업들도 같이 씀)"""
    if _KIS_KEYS.get("k"):
        return _KIS_KEYS["k"]
    try:
        k, s_ = st.secrets["KIS_APP_KEY"], st.secrets["KIS_APP_SECRET"]
        k, s_ = "".join(str(k).split()), "".join(str(s_).split())
        if k and s_:
            _KIS_KEYS["k"] = (k, s_)
            return _KIS_KEYS["k"]
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def _kis_tok_store():
    """접근토큰 보관 (하루 유효). 자주 새로 받으면 제한에 걸리므로 서버에 보관해 재사용"""
    return {"tok": None, "exp": 0.0, "lock": threading.Lock()}


def _kis_token(keys) -> str:
    import requests
    t = _kis_tok_store()
    with t["lock"]:
        if t["tok"] and time.time() < t["exp"]:
            return t["tok"]
        r = requests.post(f"{KIS_BASE}/oauth2/tokenP", timeout=10,
                          json={"grant_type": "client_credentials", "appkey": keys[0], "appsecret": keys[1]})
        j = r.json()
        if "access_token" not in j:
            raise RuntimeError(f"한국투자 토큰 실패: {j.get('error_description') or j.get('msg1') or r.status_code}")
        t["tok"], t["exp"] = j["access_token"], time.time() + int(j.get("expires_in", 86400)) - 600
        return t["tok"]


def _kis_get(path: str, tr_id: str, params: dict):
    import requests
    keys = _kis_keys()
    if not keys:
        raise RuntimeError("한국투자 키 없음(Secrets)")
    for attempt in range(5):
        h = {"content-type": "application/json; charset=utf-8", "authorization": f"Bearer {_kis_token(keys)}",
             "appkey": keys[0], "appsecret": keys[1], "tr_id": tr_id, "custtype": "P"}
        j = requests.get(KIS_BASE + path, headers=h, params=params, timeout=10).json()
        if j.get("rt_cd") == "0":
            return j
        if j.get("msg_cd") == "EGW00201":            # 초당 호출 제한 → 잠깐 쉬고 다시
            time.sleep(0.3 + 0.3 * attempt + random.random() * 0.3)
            continue
        if j.get("msg_cd") in ("EGW00123", "EGW00121"):   # 토큰 만료/무효 → 새로 받기
            _kis_tok_store()["tok"] = None
            continue
        raise RuntimeError(f"한국투자 {tr_id}: {j.get('msg_cd')} {j.get('msg1')}")
    raise RuntimeError(f"한국투자 {tr_id}: 재시도 실패")


KIS_PAR = 6      # 한국투자에 동시에 보내는 질문 수 (서버가 미국이라 한 번 왕복이 느려서 여러 개를 한꺼번에)


def kis_multprice(codes) -> dict:
    """여러 종목 실시간 시세 (30종목씩 묶어서, 여러 묶음을 동시에) → {코드: 응답 한 줄}"""
    codes = list(dict.fromkeys(codes))

    def one(chunk):
        p = {}
        for k, c in enumerate(chunk, 1):
            p[f"FID_COND_MRKT_DIV_CODE_{k}"] = "J"
            p[f"FID_INPUT_ISCD_{k}"] = c
        return _kis_get("/uapi/domestic-stock/v1/quotations/intstock-multprice", "FHKST11300006", p).get("output") or []
    chunks = [codes[i:i + 30] for i in range(0, len(codes), 30)]
    out = {}
    with ThreadPoolExecutor(max_workers=KIS_PAR) as ex:
        for rows in ex.map(one, chunks):
            for r in rows:
                out[str(r.get("inter_shrn_iscd"))] = r
    return out


def kis_top30(kind: str, iscd: str):
    """한국투자 실시간 순위 30등 (kind: 'cap' 시가총액 / 'amt' 거래대금, iscd: 0001 코스피 / 1001 코스닥)"""
    if kind == "cap":
        j = _kis_get("/uapi/domestic-stock/v1/ranking/market-cap", "FHPST01740000", {
            "fid_cond_mrkt_div_code": "J", "fid_cond_scr_div_code": "20174", "fid_div_cls_code": "0",
            "fid_input_iscd": iscd, "fid_trgt_cls_code": "0", "fid_trgt_exls_cls_code": "0",
            "fid_input_price_1": "", "fid_input_price_2": "", "fid_vol_cnt": ""})
    else:
        j = _kis_get("/uapi/domestic-stock/v1/quotations/volume-rank", "FHPST01710000", {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_COND_SCR_DIV_CODE": "20171", "FID_INPUT_ISCD": iscd,
            "FID_DIV_CLS_CODE": "0", "FID_BLNG_CLS_CODE": "3", "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "000000", "FID_INPUT_PRICE_1": "", "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "", "FID_INPUT_DATE_1": ""})
    return j.get("output") or []


POOL_N = 150     # 후보: 전날 기준 시가총액·거래대금 상위 150개씩 (+ ETF) → 실시간 시세로 다시 줄 세움


@st.cache_data(ttl=1800, show_spinner=False)
def krx_pool():
    """후보 목록(전날 또는 가장 최근 마감 기준). 하루에 몇 번만 바뀌므로 30분 보관"""
    errs = []
    for f in (_from_krx, _today_src(_from_fdr)):
        try:
            return f(POOL_N)
        except Exception as e:
            errs.append(f"{f.__name__}: {e}")
    raise RuntimeError(" / ".join(errs))


def _from_kis(n):
    """한국투자증권 실시간: 후보(전날 상위 + 한국투자 실시간 30등) 전체의 지금 시세로 1~n위를 다시 매김"""
    if not _kis_keys():
        raise RuntimeError("한국투자 키 없음(Secrets)")
    try:
        pool, _ = krx_pool()
    except Exception:
        pool = {"cap": {"KOSPI": [], "KOSDAQ": []}, "amt": {"KOSPI": [], "KOSDAQ": []}}
    out = {"cap": {}, "amt": {}}
    cands = {}
    jobs = [(k, i) for i in ("0001", "1001") for k in ("cap", "amt")]
    with ThreadPoolExecutor(max_workers=4) as ex:          # 한국투자 실시간 30등 4가지를 동시에
        top30 = dict(zip(jobs, ex.map(lambda a: kis_top30(*a), jobs)))
    for m, iscd in (("KOSPI", "0001"), ("KOSDAQ", "1001")):
        c = {}                                     # 코드 → [이름, 기준일 종가, 기준일 시가총액, 실시간 시가총액, 시가총액 후보?]
        for r in pool["cap"].get(m, []):
            c[r[0]] = [r[1], r[2], r[9], None, True]
        for r in pool["amt"].get(m, []):
            c.setdefault(r[0], [r[1], r[2], r[9], None, False])
        for r in top30[("cap", iscd)]:
            e = c.setdefault(str(r.get("mksc_shrn_iscd")), [r.get("hts_kor_isnm"), None, None, None, True])
            e[3], e[4] = (_num(r.get("stck_avls")) or 0) * 1e8 or None, True     # 시가총액 단위: 억원
        for r in top30[("amt", iscd)]:
            c.setdefault(str(r.get("mksc_shrn_iscd")), [r.get("hts_kor_isnm"), None, None, None, False])
        cands[m] = c
    live = kis_multprice([k for m in cands for k in cands[m]])
    for m, c in cands.items():
        rows_cap, rows_all = [], []
        for code, (name, pclose, pcap, kcap, is_cap) in c.items():
            q = live.get(code)
            if not q:
                continue
            px_ = _num(q.get("inter2_prpr"))
            if not px_:
                continue
            cap = kcap or (pcap * px_ / pclose if pcap and pclose else None)       # 실시간 시가총액
            row = _row(code, q.get("inter_kor_isnm") or name, px_, q.get("inter2_prdy_vrss"), q.get("prdy_ctrt"),
                       q.get("inter2_oprc"), q.get("inter2_hgpr"), q.get("inter2_lwpr"), q.get("acml_tr_pbmn"), cap)
            rows_all.append(row)
            if is_cap:
                rows_cap.append(row)
        out["cap"][m] = _top_by_cap(rows_cap, n)
        out["amt"][m] = _top_by_amount(rows_all, n)
    out["vol"] = {c: _num(q.get("acml_vol")) for c, q in live.items()}     # 오늘 누적 거래량
    if not out["cap"]["KOSPI"] or not out["cap"]["KOSDAQ"]:
        raise ValueError("한국투자 시세 빈 목록")
    return out, None


def _pool_src(n):
    """후보 목록(전날 기준)을 그대로 n위까지 사용 (한국투자 키가 없거나 실패했을 때)"""
    pool, basis = krx_pool()
    return {k: {m: v[:n] for m, v in d.items()} for k, d in pool.items()}, basis


@st.cache_data(ttl=60, show_spinner=False)
def top_lists(n: int = TOP_N):
    """{"cap": 시가총액 1~n위, "amt": 거래대금 1~n위(코스피는 ETF 포함)} × 코스피·코스닥.
    1분 동안은 저장해 둔 목록을 그대로 사용 (새로고침 버튼으로 바로 다시 받기)"""
    asof = pd.Timestamp.now(tz="Asia/Seoul").strftime("%m-%d %H:%M")
    errs = []
    # ① 한국투자증권 실시간 → ② 거래소 전날 순위(후보 목록) → ③ 네이버
    for f in (_from_kis, _pool_src, _today_src(_from_naver)):
        try:
            out, basis = f(n)
            return out, asof, basis
        except Exception as e:
            errs.append(f"{f.__name__}: {e}")
    raise RuntimeError(" / ".join(errs))           # 실패한 결과는 10분 저장하지 않고, 다음에 열 때 다시 받음


# ───────────── 공통: 시세 받기 + 매도선·매수선 계산 (리스트 신호와 차트가 똑같이 사용) ─────────────


def _start(days: int) -> str:
    return (pd.Timestamp.today() - pd.Timedelta(days=int(days * 1.6) + 10)).strftime("%Y-%m-%d")


STORE_TTL = 600     # 받은 시세를 10분 동안 재사용


@st.cache_resource(show_spinner=False)
def _store():
    """리스트에서 받은 종목 시세를 차트도 그대로 쓰도록 서버에 보관하는 저장소"""
    return {"lock": threading.Lock(), "data": {}}


@st.cache_resource(show_spinner=False)
def _sig_store():
    """종목별 신호 보관함 {(코드, 기간): (그 시세를 받은 시각, 신호)}"""
    return {}


def _hist_time(code: str, days: int):
    ent = _store()["data"].get((code, int(days)))
    return ent[0] if ent and time.time() - ent[0] < STORE_TTL else None


def known_signals(codes: tuple, days: int) -> dict:
    """시세도 신호도 이미 보관돼 있는 종목만 {코드: 신호} (받기·계산 없이 바로)"""
    out, ss_ = {}, _sig_store()
    for c in codes:
        t = _hist_time(c, days)
        ent = ss_.get((c, int(days)))
        if t is not None and ent and ent[0] == t:
            out[c] = ent[1]
    return out


def cached_hist(code: str, days: int):
    """저장소에 10분 이내에 받은 시세가 있으면 그것을, 없으면 None"""
    ent = _store()["data"].get((code, int(days)))
    if ent and time.time() - ent[0] < STORE_TTL:
        return ent[1]
    return None


def _fetch(code: str, days: int):
    d = fdr.DataReader(code, _start(days)).tail(int(days))
    st_ = _store()
    with st_["lock"]:
        st_["data"][(code, int(days))] = (time.time(), d)
    return d


def hist(code: str, days: int):
    """종목 일봉 (10분 동안 저장해 두고 리스트·차트가 같이 사용)"""
    d = cached_hist(code, days)
    return d if d is not None else _fetch(code, days)


@st.cache_data(ttl=600, show_spinner=False)
def idx_hist(sym: str, days: int):
    return fdr.DataReader(sym, _start(days)).tail(int(days))


@st.cache_resource(show_spinner=False, max_entries=32)
def _hp_inv(n: int, lam: float) -> np.ndarray:
    """HP필터 계산용 행렬의 역행렬. 기간(n)이 같으면 모든 종목이 같은 행렬이라 한 번만 만들어
    서버에 계속 보관 (화면을 새로 열어도 다시 만들지 않음)"""
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = [1, -2, 1]
    return np.linalg.inv(np.eye(n) + lam * D.T @ D)


def hp_trend(z: np.ndarray, lam: float) -> np.ndarray:
    """Hodrick-Prescott 필터의 추세 성분: (I + λ D'D) τ = z"""
    return _hp_inv(len(z), float(lam)) @ z


def calc_bands(y: np.ndarray):
    """곡선추세, 매도선(상한), 매수선(하한)"""
    t = np.arange(len(y))
    b1, b0 = np.polyfit(t, y, 1)              # 직선추세: 계산기간 전체 1차 회귀
    line = b0 + b1 * t
    curve = line + hp_trend(y - line, lam)    # 곡선추세: 직선추세 제거 → HP필터 → 복원
    width = band_pct / 100 if band_pct > 0 else kmul * (y / curve - 1).std()
    return curve, curve * (1 + width), curve * (1 - width)


TOUCH_TOL = 0          # 0 = 매도선·매수선에 정확히 닿거나 넘었을 때만 신호


def signal_of(y: np.ndarray):
    """'sell'(매도 가능) / 'buy'(매수 가능) / None"""
    _, up, lo = calc_bands(y)
    last = y[-1]
    if last >= up[-1] * (1 - TOUCH_TOL):
        return "sell"
    if last <= lo[-1] * (1 + TOUCH_TOL):
        return "buy"
    return None


def prefetch(codes: tuple, days: int, timeout: float = 40):
    """저장소에 없는(또는 10분 지난) 종목만 동시에 받아 저장. 전체 40초가 지나면 받은 것까지만 사용"""
    from concurrent.futures import wait
    missing = [c for c in codes if cached_hist(c, days) is None]
    if missing:
        ex = ThreadPoolExecutor(max_workers=8)
        futs = [ex.submit(_fetch, c, days) for c in missing]
        wait(futs, timeout=timeout)
        ex.shutdown(wait=False, cancel_futures=True)
    return {c: cached_hist(c, days) for c in codes}


def fmt_sec(x: float) -> str:
    return f"{x:.1f}초" if x < 10 else f"{x:.0f}초"


def timing_html(parts) -> str:
    """[('리스트', 0.3), ('시세', 5.2), ...] → 작은 회색 글씨 한 줄. 느린 칸(3초 이상)은 주황색"""
    out = []
    for k, v in parts:
        col = "#d9480f" if v >= 3 else "#888"
        out.append(f'{k} <b style="color:{col}">{fmt_sec(v)}</b>')
    return '<span class="tm">⏱ ' + " · ".join(out) + "</span>"


def live_px(code: str, days: int):
    """받아 둔 일봉(10분 보관)의 마지막 날 → (현재가, 대비, 등락률, 시가, 고가, 저가). 없으면 None"""
    d = cached_hist(code, days)
    if d is None or len(d) < 2:
        return None
    last, prev = d.iloc[-1], d.iloc[-2]
    c, pc = _num(last.get("Close")), _num(prev.get("Close"))
    if not c or not pc:
        return None
    return c, c - pc, (c - pc) / pc * 100, _num(last.get("Open")), _num(last.get("High")), _num(last.get("Low"))


BASE_TTL = 6 * 3600     # 신호 계산용 지난 일봉은 6시간 보관 (바뀌는 건 오늘 하루치뿐이라)


def cached_hist_any(code: str, days: int, max_age: float = BASE_TTL):
    ent = _store()["data"].get((code, int(days)))
    if ent and time.time() - ent[0] < max_age:
        return ent[1]
    return None


def prefetch_base(codes, days: int, timeout: float = 40):
    """보관함에 없거나 6시간 지난 종목만 동시에 받아 저장"""
    from concurrent.futures import wait
    missing = [c for c in dict.fromkeys(codes) if cached_hist_any(c, days) is None]
    if missing:
        ex = ThreadPoolExecutor(max_workers=8)
        futs = [ex.submit(_fetch, c, days) for c in missing]
        wait(futs, timeout=timeout)
        ex.shutdown(wait=False, cancel_futures=True)
    return len(missing)


def merged_close(code: str, days: int, live_px=None, live_vol=None):
    """보관한 지난 일봉 종가 + 오늘 실시간 가격 → 신호 계산용 종가 배열 (차트와 같은 길이)"""
    d = cached_hist_any(code, days)
    if d is None or not len(d):
        return None
    y = d["Close"].to_numpy(dtype=float).copy()
    if not live_px:
        return y
    today = pd.Timestamp.now(tz="Asia/Seoul").date()
    last_day = pd.Timestamp(d.index[-1]).date()
    last_vol = _num(d["Volume"].iloc[-1]) if "Volume" in d.columns else None
    if last_day == today:                                     # 일봉에 오늘 봉이 있음 → 지금 가격으로 바꿈
        y[-1] = live_px
    elif live_vol is not None and last_vol is not None and abs(live_vol - last_vol) < 1 and abs(live_px - y[-1]) < 1e-9:
        pass                                                  # 실시간 값 = 일봉 마지막 날 (장 시작 전·휴장일)
    elif pd.Timestamp(today).weekday() < 5:                   # 오늘 봉이 아직 없음 → 지금 가격을 오늘 봉으로 붙임
        y = np.append(y[1:] if len(y) >= int(days) else y, live_px)
    return y


@st.cache_resource(show_spinner=False)
def _warm_state():
    return {"busy": False, "lock": threading.Lock()}


def prewarm(codes, days: int):
    """다른 탭(코스피/코스닥 · 시가총액/거래대금) 종목들의 일봉을 뒤에서 미리 받아 둠 (화면은 기다리지 않음)"""
    w = _warm_state()
    with w["lock"]:
        if w["busy"]:
            return
        w["busy"] = True

    def run():
        try:
            prefetch_base(tuple(codes), days, timeout=180)
        finally:
            w["busy"] = False
    threading.Thread(target=run, daemon=True).start()


def quick_signals(codes, days: int, live: dict) -> dict:
    """일봉이 이미 보관돼 있는 종목만 바로 신호 계산 {코드: 'buy'/'sell'/None}"""
    out = {}
    for c in codes:
        y = merged_close(c, days, *live.get(c, (None, None)))
        if y is None:
            continue
        try:
            out[c] = signal_of(y) if len(y) >= 60 and np.isfinite(y).all() else None
        except Exception:
            out[c] = None
    return out


def list_signals_live(codes: tuple, live: dict):
    """신호: 지난 일봉(6시간 보관, 없는 것만 받음) + 오늘 실시간 가격"""
    days = max(int(calc), 180)
    _t = time.time()
    prefetch_base(codes, days)
    t_fetch = time.time() - _t
    _t = time.time()
    q = quick_signals(codes, days, live)
    sig = {c: q.get(c) for c in codes}
    return sig, len(q), t_fetch, time.time() - _t


def list_signals(codes: tuple):
    """리스트 종목들의 매수·매도 신호 {코드: 'buy'/'sell'/None}, 시세를 받은 종목 수"""
    days = max(int(calc), 180)
    _t = time.time()
    data = prefetch(codes, days)
    t_fetch = time.time() - _t
    _t = time.time()
    sig, got = {}, 0
    known = known_signals(codes, days)
    ss_ = _sig_store()
    for c in codes:
        d = data.get(c)
        sig[c] = None
        if d is None:
            continue
        got += 1
        if c in known:                    # 같은 시세로 이미 계산해 둔 신호
            sig[c] = known[c]
            continue
        try:
            y = d["Close"].to_numpy(dtype=float)
            if len(y) >= 60 and np.isfinite(y).all():
                sig[c] = signal_of(y)
        except Exception:
            pass
        t = _hist_time(c, days)
        if t is not None:
            ss_[(c, int(days))] = (t, sig[c])
    return sig, got, t_fetch, time.time() - _t


# ───────────── ① 리스트 화면 ─────────────
# 주소에 종목이 있으면(?code=...) 차트 화면, 없으면 리스트 화면
qp = st.query_params
KEEP = {k: qp.get(k) for k in ("embed", "embed_options") if qp.get(k)}   # 감싼 화면(index.html)용 주소 옵션 유지


def link(**kw) -> str:
    return "?" + urlencode({**KEEP, **kw})


def n0(v):
    return "-" if v is None else f"{v:,.0f}"


def ud(v, base):     # 전일 종가보다 높으면 빨강(up), 낮으면 파랑(dn)
    if v is None or base is None:
        return "eq"
    return "up" if v > base else ("dn" if v < base else "eq")


LIST_CSS = """<style>
.sl-top{display:flex;align-items:center;gap:8px;margin:4px 0 8px 0}
.sl-top a.tab{display:inline-block;min-width:92px;text-align:center;padding:8px 14px;font-size:16px;font-weight:700;
 border:1px solid #c9ced6;background:#f4f5f7;color:#555 !important;border-radius:8px;text-decoration:none !important}
.sl-top a.tab.on{background:#1f4fa8;border-color:#1f4fa8;color:#fff !important}
.sl-top a.tab.rf{min-width:0;font-size:14px;font-weight:600;color:#1f4fa8 !important;background:#fff;border-color:#9db4e0}
.sl-top a.tab.so{min-width:0;padding:7px 12px;font-size:14px;font-weight:600;background:#fff;color:#666 !important;border-color:#c9ced6}
.sl-top a.tab.so.on{background:#e8eefb;color:#1f4fa8 !important;border-color:#1f4fa8}
.sl-top .asof{margin-left:auto;font-size:12px;color:#888;white-space:nowrap}
.tm{font-size:12px;color:#888;white-space:nowrap}.tm b{font-weight:600}
.sl-top .brk{display:none}
@media (max-width:600px){.sl-top{flex-wrap:wrap;row-gap:6px}.sl-top .brk{display:block;flex-basis:100%;height:0}.sl-top .tm{flex:1 1 auto;font-size:12px}.sl-top .asof{font-size:12px}.sl-top a.tab.mk{order:1}.sl-top a.tab.rf{order:2}.sl-top .b1{order:3}.sl-top a.tab.so{order:4}.sl-top .b2{order:5}.sl-top .tm{order:6}.sl-top .asof{order:7}}
.sl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid #c9ced6}
.sl-wrap table{border-collapse:collapse;width:100%;min-width:840px;font-size:13px;margin:0}
.sl-wrap th{background:#e9edf3;color:#333;font-weight:600;padding:6px;border:1px solid #d3d8df;white-space:nowrap;text-align:center}
.sl-wrap td{padding:0;border:1px solid #e3e6ea;white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}
.sl-wrap td a{display:block;padding:6px;color:inherit !important;text-decoration:none !important}
.sl-wrap td.rk{text-align:center;color:#888;width:34px}
.sl-wrap td.cd{text-align:left;color:#333}
.sl-wrap td.nm{text-align:left;font-weight:600;color:#111;max-width:190px;overflow:hidden;text-overflow:ellipsis}
.sl-wrap td.nm .sub{display:none;font-size:11px;font-weight:400;color:#999}
.sl-wrap tr:nth-child(even) td{background:#fafbfc}
.sl-wrap tbody tr:hover td{background:#eef4ff}
.sl-wrap td.sg{text-align:center;font-weight:800;min-width:76px}.sl-wrap td.sg.buy{color:#e5484d}.sl-wrap td.sg.sell{color:#1a56db}.sl-wrap td.sg.wait{color:#bbb;font-weight:400}
.sl-wrap .up{color:#e5484d}.sl-wrap .dn{color:#1a56db}.sl-wrap .eq{color:#333}
@media (max-width:600px){
 .sl-top a.tab{min-width:80px;padding:7px 10px;font-size:15px}
 .sl-top a.tab.rf{min-width:0;padding:7px 8px;font-size:13px}
 .sl-wrap table{min-width:700px}
 .sl-wrap th.cd,.sl-wrap td.cd{display:none}
 .sl-wrap td.nm .sub{display:block}
 .sl-wrap td.nm{max-width:118px}
 .sl-wrap th.rk,.sl-wrap td.rk{position:sticky;left:0;z-index:2;background:#fff}
 .sl-wrap th.nm,.sl-wrap td.nm{position:sticky;left:34px;z-index:2;background:#fff;box-shadow:1px 0 0 #d3d8df}
 .sl-wrap th.rk,.sl-wrap th.nm{background:#e9edf3}
 .sl-wrap tr:nth-child(even) td.rk,.sl-wrap tr:nth-child(even) td.nm{background:#fafbfc}
}
</style>"""


if not qp.get("code"):
    market = "KOSDAQ" if qp.get("m") == "KOSDAQ" else "KOSPI"     # 기본값은 코스피
    sort = "amt" if qp.get("s") == "amt" else "cap"               # 기본값은 시가총액
    if qp.get("refresh"):                                         # ↻ 새로고침을 눌렀음 → 시세 다시 받기
        top_lists.clear()
        idx_hist.clear()
        with _store()["lock"]:
            _store()["data"].clear()
        del st.query_params["refresh"]
    with st.spinner("코스피·코스닥 순위를 불러오는 중입니다…"):
        _tl = time.time()
        try:
            lists, asof, basis = top_lists()
        except Exception as e:
            st.error("시세 목록을 받지 못했습니다. 1~2분 뒤에 ↻ 새로고침해 주세요.")
            st.markdown(f'<a href="{link(m=market, s=sort, refresh=1)}" target="_self">↻ 새로고침</a>', unsafe_allow_html=True)
            st.caption(f"(자세한 내용: {e})")
            st.stop()
        t_list = time.time() - _tl
    rows = lists.get(sort, {}).get(market, [])
    if not rows:
        st.error("종목 리스트를 불러오지 못했습니다. 잠시 후 새로고침해 주세요.")
        st.stop()
    def build(sig, tm=""):
        h = [LIST_CSS, '<div class="sl-top">']
        for m, label in (("KOSPI", "코스피"), ("KOSDAQ", "코스닥")):
            h.append(f'<a class="tab mk{" on" if m == market else ""}" href="{link(m=m, s=sort)}" target="_self">{label}</a>')
        for so, label in (("cap", "시가총액"), ("amt", "거래대금")):
            h.append(f'<a class="tab so{" on" if so == sort else ""}" href="{link(m=market, s=so)}" target="_self">{label}</a>')
        h.append(f'<a class="tab rf" href="{link(m=market, s=sort, refresh=1)}" target="_self" title="시세 다시 받기">↻ 새로고침</a>'
                 '<span class="brk b1"></span><span class="brk b2"></span>')
        h.append(tm or '<span class="tm">⏱ 측정 중…</span>')
        h.append(f'<span class="asof">{"순위 " + basis + " 기준 · " if basis else ""}기준 {asof}</span></div>')
        h.append('<div class="sl-wrap"><table><thead><tr><th class="rk">순위</th><th class="cd">종목코드</th>'
                 '<th class="nm">종목명</th><th>신호</th><th>현재가</th><th>대비</th><th>등락률</th>'
                 + ('<th>거래대금(억)</th>' if sort == "amt" else '<th>시가총액(억)</th>') +
                 '<th>시가</th><th>고가</th><th>저가</th></tr></thead><tbody>')
        for i, (c, nm, close, chg, pct, op, hi, lo, amt, cap) in enumerate(rows, 1):
            base = amt if sort == "amt" else cap               # 순위 기준 칸에 보여줄 값
            if basis or op is None:                             # 전날 순위일 때만: 현재가·대비·시가·고가·저가를 받아 둔 일봉(실시간)으로
                _lp = live_px(c, days)
                if _lp:
                    close, chg, pct, op, hi, lo = _lp
            href = html.escape(link(code=c, name=nm, mkt=market, s=sort))
            a = f'<a href="{href}" target="_self">'
            prev = close - chg if (close is not None and chg is not None) else None
            k = "eq" if pct is None else ("up" if pct > 0 else ("dn" if pct < 0 else "eq"))
            arrow = "" if not pct else ("▲ " if pct > 0 else "▼ ")
            pct_t = "-" if pct is None else f"{pct:+.2f}%" if pct else "0.00%"
            nm_e, c_e = html.escape(nm), html.escape(c)
            if sig is None or c not in sig:                   # 아직 계산 중
                sg = '<td class="sg wait">' + a + '…</a></td>'
            else:
                v = sig.get(c)
                sg = ('<td class="sg buy">' + a + '매수 가능</a></td>' if v == "buy" else
                      '<td class="sg sell">' + a + '매도 가능</a></td>' if v == "sell" else
                      '<td class="sg">' + a + '&nbsp;</a></td>')
            h.append(
                f'<tr><td class="rk">{a}{i}</a></td><td class="cd">{a}{c_e}</a></td>'
                f'<td class="nm">{a}{nm_e}<span class="sub">{c_e}</span></a></td>'
                f'{sg}'
                f'<td class="{k}">{a}{n0(close)}</a></td>'
                f'<td class="{k}">{a}{arrow}{n0(None if chg is None else abs(chg))}</a></td>'
                f'<td class="{k}">{a}{pct_t}</a></td>'
                f'<td class="eq">{a}{"-" if base is None else f"{base / 1e8:,.0f}"}</a></td>'
                f'<td class="{ud(op, prev)}">{a}{n0(op)}</a></td>'
                f'<td class="{ud(hi, prev)}">{a}{n0(hi)}</a></td>'
                f'<td class="{ud(lo, prev)}">{a}{n0(lo)}</a></td></tr>')
        h.append('</tbody></table></div>')
        return "".join(h)                             # 줄바꿈 없이 한 덩어리로 넣어야 표가 깨지지 않음

    codes = tuple(r[0] for r in rows)
    days = max(int(calc), 180)
    note = st.empty()
    ph = st.empty()
    _vol = lists.get("vol") or {}
    live = {r[0]: (r[2], _vol.get(r[0])) for r in rows} if (basis is None and _vol) else {}   # 오늘 실시간 가격
    _tq = time.time()
    known = quick_signals(codes, days, live)                  # 일봉이 보관된 종목은 받기 없이 바로 신호
    t_quick = time.time() - _tq
    _t0 = time.time()
    if len(known) == len(codes):                              # 전부 보관돼 있으면 한 번에 표시
        sig, _got, t_fetch, t_calc = dict(known), len(codes), 0.0, t_quick
    else:
        ph.markdown(build(known), unsafe_allow_html=True)     # ① 리스트를 먼저 표시
        note.caption("⏳ 매수·매도 신호를 계산하는 중입니다… (처음 한 번만 조금 걸려요)")
    try:
        if len(known) < len(codes):
            sig, _got, t_fetch, t_calc = list_signals_live(codes, live)
    except Exception as e:
        sig, _got, t_fetch, t_calc = {}, len(codes), time.time() - _t0, 0.0
        st.caption(f"신호 계산 오류: {e}")
    note.empty()
    if _got < len(rows):          # 시세를 다 못 받았으면 알려줌 (빈칸 = 계산 못 한 종목 포함)
        st.caption(f"※ {len(rows)}종목 중 {_got}종목만 시세를 받아 신호를 계산했습니다 ({time.time() - _t0:.0f}초).")
    _tm = timing_html([("리스트", t_list), ("시세", t_fetch), ("계산", t_calc)])
    ph.markdown(build(sig, _tm), unsafe_allow_html=True)      # ② 신호 칸 + 걸린 시간 채워서 표시
    try:                                                      # 다른 탭 종목들의 일봉은 뒤에서 미리 받아 둠
        prewarm([r[0] for so in ("cap", "amt") for mk in ("KOSPI", "KOSDAQ") for r in lists.get(so, {}).get(mk, [])], days)
    except Exception:
        pass
    st.stop()


# ───────────── ② 차트 화면 (주식001 차트) ─────────────
_t_chart0 = time.time()
_back_m = "KOSDAQ" if qp.get("mkt") == "KOSDAQ" else "KOSPI"
_back_s = "amt" if qp.get("s") == "amt" else "cap"
BACK_CSS = ('<style>.cv-top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:2px 0 4px 0}'
            '.cv-top a.back{display:inline-block;padding:4px 14px;font-size:15px;font-weight:600;border:1px solid #c9ced6;'
            'border-radius:6px;background:#f4f5f7;color:#333 !important;text-decoration:none !important}'
            '.tm{font-size:12px;color:#888;white-space:nowrap}.tm b{font-weight:600}</style>')


def top_bar(tm: str) -> str:
    """← 목록으로 버튼 + 걸린 시간 (돌아가면 보던 탭이 나오게 m=코스피/코스닥)"""
    return (BACK_CSS + '<div class="cv-top">'
            f'<a class="back" href="{html.escape(link(m=_back_m, s=_back_s))}" target="_self">← 목록으로</a>{tm}</div>')


top_ph = st.empty()
top_ph.markdown(top_bar('<span class="tm">⏱ 측정 중…</span>'), unsafe_allow_html=True)

code = str(qp.get("code")).strip()
name = str(qp.get("name") or code)
mkt = "KOSDAQ" if qp.get("mkt") == "KOSDAQ" else "KOSPI"
name_ph = st.empty()
name_ph.markdown(
    f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
    f'<span style="font-size:13px;color:#888;font-weight:400">{html.escape(code)}</span></div>',
    unsafe_allow_html=True)

# 시세를 받는 동안 "고장난 것처럼" 보이지 않도록 안내 표시 (차트가 나오면 사라짐)
status = st.empty()
if cached_hist(code, max(int(calc), 180)) is None:
    status.markdown(
        '<div style="padding:10px 12px;border-radius:8px;background:#fff4dc;color:#8a5a00;font-size:14px;line-height:1.5">'
        '⏳ 시세를 불러오는 중입니다. 처음 한 번은 20~30초 걸릴 수 있어요. 잠시만 기다려 주세요.</div>',
        unsafe_allow_html=True)

@st.cache_data(ttl=30, show_spinner=False)
def live_index(sym: str):
    """네이버에서 지수 현재값(장중이면 실시간, 장 마감 후엔 종가)을 받는다. 실패하면 None"""
    import requests
    try:
        r = requests.get(f"https://m.stock.naver.com/api/index/{'KOSDAQ' if sym == 'KQ11' else 'KOSPI'}/basic",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        j = r.json()
        val = float(str(j["closePrice"]).replace(",", ""))
        day = pd.Timestamp(str(j.get("localTradedAt", ""))[:10]).normalize()
        return day, val
    except Exception:
        return None


def load_both(code: str, idx_sym: str, days: int):
    """종목과 지수 시세를 동시에 받아 온다. 지수는 8초 안에 못 받으면 None(차트는 지수선 없이 표시)"""
    ex = ThreadPoolExecutor(max_workers=2)
    f_stock = ex.submit(hist, code, int(days))
    f_idx = ex.submit(idx_hist, idx_sym, int(days))
    try:
        stock = f_stock.result(timeout=40)
        try:
            idx = f_idx.result(timeout=8)
        except Exception:
            idx = None
        return stock, idx
    finally:
        ex.shutdown(wait=False)


calc = max(int(calc), 180)
idx_sym, idx_name = ("KQ11", "코스닥") if mkt == "KOSDAQ" else ("KS11", "코스피")
try:
    _tf = time.time()
    df, _idf = load_both(code.strip(), idx_sym, calc)
    t_fetch = time.time() - _tf
except Exception as e:
    status.empty()
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()
if df.empty or len(df) < 60:
    status.empty()
    st.warning("데이터가 부족합니다(60개 이상 필요). 종목코드를 확인하세요.")
    st.stop()

_tc = time.time()
y = df["Close"].to_numpy(dtype=float)
n = len(y)
t = np.arange(n)

# 곡선추세·매도선·매수선 (리스트 신호와 같은 계산)
curve, upper, lower = calc_bands(y)

# 화면에는 최근 disp일만 표시
D = n
init_span = int(min(disp, D))
sl = slice(n - D, n)
dates = df.index.strftime("%Y-%m-%d")[sl]
yv, curvev, upv, lov = y[sl], curve[sl], upper[sl], lower[sl]
def _ohl(c):   # 시가·고가·저가가 비어 있으면 종가로 대신 채움
    a = df[c].to_numpy(dtype=float)[sl] if c in df.columns else yv.copy()
    return np.where(np.isfinite(a) & (a > 0), a, yv)
ov, hv, lv = _ohl("Open"), _ohl("High"), _ohl("Low")
hv, lv = np.maximum.reduce([hv, ov, yv]), np.minimum.reduce([lv, ov, yv])

# ───────────── 지수(코스피/코스닥) 비교선 ─────────────
idx_arr = None
_live_ok = False
try:
    _s = _idf["Close"].reindex(df.index).ffill().bfill()
    _live = live_index(idx_sym)                 # 종목처럼 접속할 때마다 최신 지수값으로 갱신
    if _live is not None and _live[0] >= df.index[-1].normalize():
        _s.iloc[-1] = _live[1]
        _live_ok = True
    if len(_s) == n and np.isfinite(_s.to_numpy(dtype=float)).all():
        idx_arr = _s.to_numpy(dtype=float)
except Exception:
    idx_arr = None
idx_note = ""
if idx_arr is not None:     # 지수 데이터가 종목보다 며칠 일찍 끝나면 알려줌 (마지막 값으로 채워졌다는 뜻)
    _gap = (df.index.max() - _idf.index.max()).days
    if _gap >= 1 and not _live_ok:
        idx_note = f"※ {idx_name} 데이터가 {_idf.index.max():%Y-%m-%d}까지만 있어 최근 {_gap}일은 마지막 값으로 채웠습니다."
idx_scaled = None
if idx_arr is not None:     # 처음 열었을 때 맨 왼쪽 종가에 한 번만 맞춘 뒤 고정 (밀거나 확대해도 변하지 않음)
    idx_scaled = idx_arr * (yv[n - init_span] / idx_arr[n - init_span])
PHONE_SPAN_PY = min(42, init_span)
idx_scaled_p = None
if idx_scaled is not None:  # 폰은 처음 2달치만 보이므로 그 맨 왼쪽 종가에 맞춘 지수선도 따로 준비
    idx_scaled_p = idx_arr * (yv[n - PHONE_SPAN_PY] / idx_arr[n - PHONE_SPAN_PY])

# ───────────── 상단 시세 헤더 ─────────────
last, prev = y[-1], y[-2]
chg = last - prev
pct = chg / prev * 100
col = "#e5484d" if chg > 0 else ("#3b6fd4" if chg < 0 else "#333333")
arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "-")
vol = df["Volume"].iloc[-1]

# ───────────── 종목명 옆: 기준 날짜 + 지수 ─────────────
if idx_arr is not None:
    name_ph.markdown(
        f'<div style="display:flex;align-items:baseline;gap:18px;flex-wrap:wrap">'
        f'<div style="font-size:20px;font-weight:700;line-height:1.3">{html.escape(str(name))} '
        f'<span style="font-size:13px;color:#888;font-weight:400">{html.escape(code)}</span></div>'
        f'<div style="font-size:17px;font-weight:700;color:#0aa05a">'
        f'● {df.index[-1]:%Y-%m-%d} {idx_name} {idx_arr[-1]:,.2f}</div>'
        '</div>',
        unsafe_allow_html=True)

# ───────────── 매수·매도 표시 (현재가가 곡선추세 상·하한에 닿거나 넘었을 때) ─────────────
_up, _lo = float(upper[-1]), float(lower[-1])
sig_html = ""
if last >= _up * (1 - TOUCH_TOL):          # 상한에 닿거나 위로 넘음 → 파란색 매도
    sig_html = '<div style="font-size:40px;font-weight:800;color:#1a56db;line-height:1">매도</div>'
elif last <= _lo * (1 + TOUCH_TOL):        # 하한에 닿거나 아래로 내려감 → 빨간색 매수
    sig_html = '<div style="font-size:40px;font-weight:800;color:#e5484d;line-height:1">매수</div>'

_sig = ""
if sig_html:
    _sig_txt, _sig_col = ("매도 가능합니다", "#1a56db") if "매도" in sig_html else ("매수 가능합니다", "#e5484d")
    _sig = f'<div class="hd-sig" style="color:{_sig_col}">{_sig_txt}</div>'
st.markdown(
    '<style>'
    '.hd-row{display:flex;align-items:center;gap:16px;flex-wrap:nowrap}'
    '.hd-px{display:flex;flex-direction:column;align-items:flex-start;gap:4px}'
    '.hd-big{font-size:42px;font-weight:700;line-height:1}'
    '.hd-chg{font-size:16px}'
    '.hd-lines{display:flex;flex-direction:column;gap:10px}'
    '.hd-lbl{font-size:15px;color:#111;line-height:1.4;text-align:center;white-space:nowrap}'
    '.hd-sig{font-size:24px;font-weight:800;line-height:1.2;white-space:nowrap}'
    '.hd-wrap{display:flex;align-items:flex-end;justify-content:space-between;gap:8px 16px;flex-wrap:wrap}'
    '.cv-box{margin-left:auto;padding-right:4px;padding-bottom:2px}'
    '@media (max-width:600px){'
    '.hd-row{gap:10px}.hd-big{font-size:30px}.hd-chg{font-size:14px}'
    '.hd-lbl{font-size:12px}.hd-sig{font-size:17px}.cv-box{margin-left:0;width:100%}}'
    '</style>'
    '<div class="hd-wrap"><div class="hd-row">'
    '<div>'
    f'<div class="hd-px"><span class="hd-big" style="color:{col}">{last:,.0f}</span>'
    f'<span class="hd-chg" style="color:{col}">{arrow} {abs(chg):,.0f} &nbsp; {pct:+.2f}%</span></div>'
    f'<div style="font-size:13px;color:#888;margin-top:6px">거래량 {vol:,.0f}</div>'
    '</div>'
    '<div class="hd-lines">'
    f'<div class="hd-lbl" data-ul="1" style="color:#1a56db;font-weight:600">매도선<br>(<span data-k="u">{_up:,.0f}</span>)</div>'
    f'<div class="hd-lbl" data-ul="1" style="color:#e5484d;font-weight:600">매수선<br>(<span data-k="l">{_lo:,.0f}</span>)</div>'
    '</div>'
    f'{_sig}'
    '</div><div class="cv-box"></div></div>',      # cv-box: 차트가 지수 비교 정보·범례·버튼을 여기에 채움
    unsafe_allow_html=True)

# ───────────── 차트 ─────────────
def direction_colors(v):
    d = np.diff(v, prepend=v[1])
    d[0] = d[1]
    return np.where(d >= 0, "#e5484d", "#3b6fd4")  # 상승=빨강, 하락=파랑


def base_traces(fig):
    fig.add_trace(go.Candlestick(x=dates, open=ov, high=hv, low=lv, close=yv, name="수정종가", showlegend=False,
                                 increasing=dict(line=dict(color="#e5484d", width=1), fillcolor="#e5484d"),
                                 decreasing=dict(line=dict(color="#3b6fd4", width=1), fillcolor="#3b6fd4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=curvev, name="곡선추세", line=dict(color="#111111", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=upv, name="매도선", line=dict(color="#1a56db", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=lov, name="매수선", line=dict(color="#e5484d", width=2.2)), row=1, col=1)
    if idx_scaled is not None:   # 4번 트레이스: 지수(녹색 점선)
        fig.add_trace(go.Scatter(x=dates, y=idx_scaled, name=idx_name,
                                 line=dict(color="#00a651", width=3, dash="dot")), row=1, col=1)
        if idx_scaled_p is not None:   # 5번 트레이스: 폰용 지수선(처음엔 숨김, 폰이면 JS가 이것으로 교체)
            fig.add_trace(go.Scatter(x=dates, y=idx_scaled_p, name=idx_name, visible=False, showlegend=False,
                                     line=dict(color="#00a651", width=3, dash="dot")), row=1, col=1)


CROSSHAIR_JS = """
function installCrosshair(gd, initSpan, S, B, IDX, IT, IDXNAME, IDXRAW, OPEN, LOW, ANCH, IDX2, IT2, HIGH){
  var LP_MS=500;           // 길게 눌러 진동이 오기까지의 시간(ms)
  var CROSS_DELAY_MS=150;  // 진동 후 십자선이 나타나기까지의 시간(ms) (진동 → 십자선 순서 맞춤)
  var PAN_GAIN=1.5;        // 손가락 이동 속도 배율
  var ZOOM_GAIN=1.5;       // 확대/축소 속도 배율
  var COMPARE_MODE='level'; // 'level'=그날 종목 가격과 그날 지수선(녹색 점선) 비교, 'daily'=그날 등락률끼리 비교
  var SAME_BAND_LEVEL=2;    // level: 가격 차이가 ±2% 이내면 '보합'
  var SAME_BAND_DAILY=0.3;  // daily: 등락률 차이가 ±0.3%p 이내면 '보합'
  var idxOn=true;          // 지수 비교선 표시 여부
  var RPAD_RATIO=0.08;     // 오른쪽 여백: 보이는 기간의 8%만큼 빈 칸 (120일 → 약 10칸, 폰 42일 → 약 3칸)
  var RPAD_MIN=3;          // 오른쪽 여백 최소 칸 수
  // 브라우저(앱 브라우저)가 두 손가락으로 화면 전체를 확대하지 못하게 막기
  try{
    var pd=window.parent.document, mv=pd.querySelector('meta[name=viewport]');
    if(mv)mv.setAttribute('content','width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no');
    if(!pd.getElementById('nozoom_css')){
      var st=pd.createElement('style'); st.id='nozoom_css';
      st.textContent='iframe{touch-action:pan-y !important;}'; pd.head.appendChild(st);
    }
  }catch(err){}
  gd.style.position='relative'; gd.style.touchAction='pan-y';
  gd.style.userSelect='none'; gd.style.webkitUserSelect='none';
  gd.style.webkitTouchCallout='none'; gd.style.webkitTapHighlightColor='transparent';
  var cats=gd.data[0].x, N=cats.length, MINS=Math.min(20,N), MAXS=Math.min(Math.max(120,initSpan),N);
  function clampSpan(v){return Math.round(Math.max(MINS,Math.min(MAXS,v)));}
  function clampI0(v,sp){return Math.round(Math.max(0,Math.min(N-sp,v)));}
  // 핸드폰(화면 폭 600px 이하)은 처음에 약 2달(42거래일)만 확대해서 보여줌. PC는 그대로.
  var PHONE_SPAN=42, ITa=IT;
  var isPhone=false;
  try{ isPhone=Math.min(window.screen&&window.screen.width||9999, window.innerWidth||9999)<=600; }catch(e0){}
  if(isPhone && initSpan>PHONE_SPAN && IDX && IDX2 && IDX2.length===N){
    try{   // 폰: 2달치 기준으로 미리 환산해 둔 지수선으로 교체(보이기/숨기기만 사용)
      IDX=IDX2; ITa=IT2; initSpan=PHONE_SPAN;
      var pr0=Plotly.restyle(gd,{visible:[false]},[IT]); if(pr0&&pr0.then)pr0.then(null,function(){});
      var pr1=Plotly.restyle(gd,{visible:[true]},[IT2]); if(pr1&&pr1.then)pr1.then(null,function(){});
    }catch(e1){}
  } else if(isPhone && initSpan>PHONE_SPAN){ initSpan=PHONE_SPAN; }
  var span=clampSpan(initSpan), i0=N-span;
  var pending=false, mode=null, timer=null, sx=0, sy=0, startI0=0, lastT=null, mdown=false, pinch=null, markersOn=null;
  var tStart=0, lastTap=0, lastTouch=0;

  function mk(css){var d=document.createElement('div');
    d.style.cssText='position:absolute;pointer-events:none;display:none;z-index:50;'+css;
    gd.appendChild(d);return d;}
  var lab='background:#222;color:#fff;font:11px sans-serif;padding:2px 4px;white-space:nowrap;';
  var vl=mk('width:0;border-left:1px solid #333;');
  var hl=mk('height:0;border-top:1px solid #333;');
  var dl=mk(lab+'transform:translateX(-50%);white-space:pre;text-align:center;line-height:1.3;');   // 날짜 + (가격 , 지수) 두 줄
  var pr=mk(lab+'transform:translateY(-50%);');
  function f2(v){return Number(v).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}
  // 오른쪽 위 상자: 비교 요약 3줄 → 범례 + [지수][＋][－] 버튼
  // 위쪽 시세 줄 오른쪽 빈자리(cv-box)에 넣음. 못 찾으면 차트 오른쪽 위에 표시
  var HD=document, host=null;
  try{ host=window.parent.document.querySelector('.cv-box'); if(host){HD=window.parent.document; host.innerHTML='';} }catch(eh){host=null; HD=document;}
  // 상자 = [글씨 칸(비교 요약·범례)] + [버튼 칸]. 버튼 칸은 오른쪽 끝에 따로 고정 → 글씨 길이가 바뀌어도 버튼은 안 움직임
  var outer=HD.createElement('div');
  outer.style.cssText=(host?'':'position:absolute;top:2px;right:4px;z-index:55;')+'max-width:100%;'
    +'display:flex;align-items:flex-end;justify-content:flex-end;gap:14px;';
  (host||gd).appendChild(outer);
  var box=HD.createElement('div');
  box.style.cssText='display:flex;flex-direction:column;align-items:flex-start;gap:2px;min-width:0;';
  outer.appendChild(box);
  var bcol=HD.createElement('div');
  bcol.style.cssText='display:flex;align-items:center;gap:6px;flex:none;';
  outer.appendChild(bcol);
  function bk(css){var d=HD.createElement('div'); d.style.cssText='display:none;'+css; box.appendChild(d); return d;}
  var sm=bk('font:600 12px sans-serif;white-space:nowrap;');                 // 지수 대비 요약
  var dt=bk('font:11px sans-serif;color:#666;white-space:normal;word-break:keep-all;line-height:1.35;');   // 줄바꿈은 띄어쓰기에서만   // 시가·저가·종가·지수
  var ig=bk('font:600 11px sans-serif;color:#00a651;white-space:nowrap;');   // 지수 값(녹색)
  if(IDX){sm.style.display='block'; dt.style.display='block'; ig.style.display='block';}
  var brow=HD.createElement('div');
  brow.style.cssText='margin-top:2px;line-height:1.35;';
  brow.innerHTML='<span style="font:11px sans-serif;color:#111;white-space:nowrap;margin-right:4px">'
    +'<span style="color:#e5484d">▮</span><span style="color:#3b6fd4">▮</span> 수정종가 캔들&nbsp; ━ 곡선&nbsp; '
    +'<span style="color:#1a56db">━ 상한</span>&nbsp; <span style="color:#e5484d">━ 하한</span></span>';
  box.appendChild(brow);
  // 폰(폭 600px 이하): 글씨는 오른쪽 끝까지 넓게 쓰고, 맨 아래 줄은 [범례 ……… 버튼]으로 배치
  var narrow=false;
  try{ narrow=Math.min(window.screen&&window.screen.width||9999, (HD.defaultView||window).innerWidth||9999)<=600; }catch(en){}
  if(narrow){
    outer.style.flexDirection='column'; outer.style.alignItems='stretch'; outer.style.gap='4px';
    box.style.width='100%';
    var lastRow=HD.createElement('div');
    lastRow.style.cssText='display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;';
    outer.appendChild(lastRow);
    lastRow.appendChild(brow); lastRow.appendChild(bcol);
    brow.style.marginTop='0';
  }
  var focus=N-1;           // 십자선이 가리키는 날짜(없으면 맨 오른쪽=최근)
  function sg(v){return (v>=0?'+':'')+v.toFixed(1)+'%';}
  function md(i){return cats[i].slice(5);}
  function pxs(v){return Number(v).toLocaleString('en-US',{maximumFractionDigits:0});}
  function updateSummary(){   // 선택한 날짜(없으면 최근 날짜)의 종목 가격과 지수로 강함/약함/보합 판단
    var D0=focus;
    try{
      var su=window.parent.document.querySelectorAll('[data-k=u]'), sl2=window.parent.document.querySelectorAll('[data-k=l]');
      for(var q=0;q<su.length;q++)su[q].textContent=pxs(S[2][D0]);
      for(var q2=0;q2<sl2.length;q2++)sl2[q2].textContent=pxs(S[3][D0]);}catch(e2){}
    if(!IDX)return;
    var R=IDXRAW||IDX, D=focus, d, band, row2, unit;
    if(COMPARE_MODE==='daily' && D>=1){
      var sr=(S[0][D]/S[0][D-1]-1)*100, ir=(R[D]/R[D-1]-1)*100;
      d=sr-ir; band=SAME_BAND_DAILY; unit='%p ';
      row2='그날 등락 종목 '+sg(sr)+' · '+IDXNAME+' '+sg(ir);
    } else {
      d=(S[0][D]/IDX[D]-1)*100; band=SAME_BAND_LEVEL; unit='% ';     // 종목 가격이 지수선보다 몇 % 위/아래인지
      // 시가·저가·종가는 전일 종가 대비, 지수선은 전일 지수선 대비 등락률
      var prevClose = D > 0 ? S[0][D-1] : NaN;
      var prevIndex = D > 0 ? IDX[D-1] : NaN;
      function coloredValue(label, value, base, pctSourceValue){
        // pctSourceValue가 있으면 실제 원본 지수값으로 등락률을 계산
        var compareValue = (pctSourceValue !== undefined) ? pctSourceValue : value;
        var pct = (isFinite(base) && base !== 0 && isFinite(compareValue))
                    ? (compareValue / base - 1) * 100 : NaN;
        var color = !isFinite(pct) || Math.abs(pct) < 0.0000001 ? '#000000' : (pct > 0 ? '#e5484d' : '#3b6fd4');
        var pctText = isFinite(pct) ? (pct > 0 ? '+' : '') + pct.toFixed(2) + '%' : '—';
        return '<span style="color:'+color+'">'+label+' '+(label===IDXNAME?f2(value):pxs(value))+' ('+pctText+')</span>';
      }
      var prevRawIndex = D > 0 ? R[D-1] : NaN;
      row2 = '<span style="color:#000000">시가 '+pxs(OPEN[D])+'</span> · ' +
       '<span style="color:#000000">저가 '+pxs(LOW[D])+'</span> · ' +
       '<span style="color:#000000">종가 '+pxs(S[0][D])+'</span> · ' +
       coloredValue(IDXNAME, R[D], prevRawIndex);
    }
    var up=d>band, dn=d<-band;
    sm.textContent=IDXNAME+' 대비 '+(d>=0?'+':'')+d.toFixed(1)+unit+(up?'강함':(dn?'약함':'보합'));
    sm.style.color=up?'#e5484d':(dn?'#3b6fd4':'#666');
    if(COMPARE_MODE==='daily'){
      dt.textContent=row2;
    } else {
      dt.innerHTML=row2;
    }
    ig.textContent='● '+cats[D]+'  '+IDXNAME+' '+f2(R[D]);
  }
  var tg=null;
  var dbg=mk('left:4px;top:66px;font:11px sans-serif;color:#c00;background:rgba(255,255,255,.92);padding:2px 4px;max-width:92%;z-index:70;');
  function showErr(e){dbg.style.display='block'; dbg.textContent='차트 오류: '+(e&&e.message?e.message:e);}
  function hide(){[vl,hl,dl,pr].forEach(function(e){e.style.display='none';}); if(focus!==N-1){focus=N-1; updateSummary();}}

  function fmt(c){return c.slice(5).replace('-','.');}

  // 보이는 구간의 최고/최저에 맞춰 가격 범위를 즉시 변경 + 지수선을 맨 왼쪽 종가에 맞춰 환산
  function draw(){
    var a=Math.max(0,Math.floor(i0)), b=Math.min(N-1,Math.ceil(i0+span)-1), i, k, v;
    var lo=Infinity, hi=-Infinity, bl=0, bh=0;
    for(k=0;k<S.length;k++){for(i=a;i<=b;i++){v=S[k][i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(IDX&&idxOn){for(i=a;i<=b;i++){v=IDX[i]; if(v<lo)lo=v; if(v>hi)hi=v;}}
    if(HIGH&&LOW){for(i=a;i<=b;i++){if(HIGH[i]>hi)hi=HIGH[i]; if(LOW[i]<lo)lo=LOW[i];}}   // 캔들 고가·저가
    for(i=a;i<=b;i++){v=B[i]; if(v<bl)bl=v; if(v>bh)bh=v;}
    if(!isFinite(lo)||!isFinite(hi)||!isFinite(bl)||!isFinite(bh))throw new Error('가격 범위 계산 오류');
    var pad=(hi-lo)*0.06||1;
    var step=Math.max(3,Math.round(span/4)), tv=[], tt=[], idx;
    for(idx=N-1;idx>=a;idx-=step){tv.push(cats[idx]); tt.push(fmt(cats[idx]));}
    var rp=Math.max(RPAD_MIN,Math.round(span*RPAD_RATIO));   // 오늘 캔들이 오른쪽 끝에 붙지 않게 빈 칸 추가
    var xr=[i0-0.5,i0+span-0.5+rp];
    var lay={'xaxis.range':xr,'xaxis2.range':xr,
      'yaxis.range':[lo-pad,hi+pad],'yaxis2.range':[bl*1.1,(bh*1.1)||1],
      'xaxis.tickvals':tv,'xaxis.ticktext':tt,'xaxis2.tickvals':tv,'xaxis2.ticktext':tt};
    var p=Plotly.relayout(gd,lay);
    if(p&&p.then)p.then(null,showErr);
  }
  function render(){
    pending=false;
    try{ draw(); }
    catch(e){                                // 지수 부분에서 문제가 생겨도 차트는 계속 그리도록 지수 기능만 끔
      showErr(e);
      if(IDX){
        IDX=null; sm.style.display='none'; dt.style.display='none'; ig.style.display='none'; if(tg)tg.style.display='none';
        try{ Plotly.restyle(gd,{visible:[false]},[ITa]); }catch(e1){}
        try{ draw(); }catch(e2){ showErr(e2); }
      }
    }
  }
  function schedule(){if(!pending){pending=true;requestAnimationFrame(render);}}

  function move(cx,cy){
    var r=gd.getBoundingClientRect(), x=cx-r.left, y=cy-r.top;
    var fl=gd._fullLayout, s=fl._size, xa=fl.xaxis, ya=fl.yaxis, yb=fl.yaxis2;
    if(x<s.l||x>s.l+s.w||y<s.t||y>s.t+s.h){hide();return;}
    var lo=xa.range[0], hi=xa.range[1];
    var idx=Math.round((x-xa._offset)/xa._length*(hi-lo)+lo);
    idx=Math.max(0,Math.min(N-1,idx));
    focus=idx; updateSummary();
    var xs=xa._offset+(idx-lo)/(hi-lo)*xa._length;
    if(!isFinite(xs))return;
    var ax=(y>=yb._offset)?yb:ya;
    var v=ax.range[1]-(y-ax._offset)/ax._length*(ax.range[1]-ax.range[0]);
    var txt=Math.round(v).toLocaleString('en-US');
    vl.style.cssText+=';display:block;left:'+xs+'px;top:'+s.t+'px;height:'+s.h+'px;';
    hl.style.cssText+=';display:block;left:'+s.l+'px;top:'+y+'px;width:'+s.w+'px;';
    var showI=!!(IDX&&IDXRAW&&idxOn);
    // 예) 2026-10-20 (줄바꿈) (11,700 , 5,600.00)  → 날짜 / (그날 종목 가격 , 그날 지수)
    dl.textContent=cats[idx]+'\\n('+pxs(S[0][idx])+(showI?' , '+f2(IDXRAW[idx]):'')+')';
    dl.style.display='block'; dl.style.top=(ya._offset+ya._length+2)+'px';
    var half=(dl.offsetWidth||0)/2, tot=fl.width||gd.clientWidth||0;
    dl.style.left=(tot?Math.max(half,Math.min(tot-half,xs)):xs)+'px';        // 화면 밖으로 나가지 않게
    pr.style.display='block'; pr.style.left=(s.l+s.w+2)+'px'; pr.style.top=y+'px'; pr.textContent=txt;
  }
  function pxToBars(dx){return dx/gd._fullLayout._size.w*span;}
  function fracOf(clientX){
    var r=gd.getBoundingClientRect(), s=gd._fullLayout._size;
    return Math.max(0,Math.min(1,(clientX-r.left-s.l)/s.w));
  }
  function dist(t){var dx=t[0].clientX-t[1].clientX, dy=t[0].clientY-t[1].clientY; return Math.sqrt(dx*dx+dy*dy);}
  function resetView(){                       // 더블클릭/더블탭: 처음 상태로
    span=clampSpan(initSpan); i0=N-span; pinch=null; if(mode==='cross')mode=null; hide(); schedule();
  }
  function zoomBy(factor){                    // 화면 가운데 기준 확대/축소
    var c=i0+span/2, ns=clampSpan(span*factor);
    if(ns===span)ns=clampSpan(span+(factor<1?-1:1));
    span=ns; i0=clampI0(c-ns/2,ns); hide(); schedule();
  }

  // ── 확대/축소 버튼 (두 손가락 확대가 막히는 폰용) ──
  function btn(txt,fn,w,fs){
    var b=HD.createElement('div'); b.textContent=txt;
    b.style.cssText=(w?'padding:0 10px;':'width:32px;')+'height:28px;line-height:28px;text-align:center;white-space:nowrap;'
      +'font:600 '+(fs||16)+'px sans-serif;background:#f3f3f3;border:1px solid #ccc;border-radius:5px;cursor:pointer;'
      +'touch-action:manipulation;user-select:none;-webkit-user-select:none;';
    ['touchstart','touchmove','touchend','mousedown','mousemove','dblclick'].forEach(function(n){
      b.addEventListener(n,function(e){e.stopPropagation();},{passive:true});});
    b.addEventListener('click',function(e){e.stopPropagation();fn();});
    bcol.appendChild(b);
    return b;
  }
  if(IDX){                                  // 비교선 켜기/끄기 버튼 (버튼 글자 = 지수 이름)
    tg=btn(IDXNAME,function(){idxOn=!idxOn; paintTg(); try{Plotly.restyle(gd,{visible:[idxOn]},[ITa]);}catch(e){showErr(e);} schedule();},1,14);
    var paintTg=function(){tg.style.color=idxOn?'#007a3d':'#999'; tg.style.background=idxOn?'#e8f8ef':'#f3f3f3'; tg.style.borderColor=idxOn?'#00a651':'#ccc';};
    paintTg();
  }
  btn('＋',function(){zoomBy(0.6);});
  btn('－',function(){zoomBy(1/0.6);});


  // ── 손가락 ──
  // 바로 좌우로 밀기=기간 이동(한 칸씩 딱딱하게) / 길게 눌러 진동=십자선 / 두 손가락=확대·축소
  gd.addEventListener('touchstart',function(e){
    clearTimeout(timer); lastTouch=Date.now();
    if(e.touches.length>=2){
      hide(); mode='pinch';
      pinch={d:dist(e.touches),span:span,anchor:i0+fracOf((e.touches[0].clientX+e.touches[1].clientX)/2)*span};
      return;
    }
    hide();
    var t=e.touches[0]; lastT=t; sx=t.clientX; sy=t.clientY; startI0=i0; mode=null; tStart=Date.now();
    timer=setTimeout(function(){
      if(mode!==null)return;
      mode='arming'; if(navigator.vibrate)navigator.vibrate(40);       // ① 진동
      timer=setTimeout(function(){                                      // ② 잠깐 뒤 십자선 표시 후 따라다님
        if(mode==='arming'){mode='cross'; move(lastT.clientX,lastT.clientY);}
      },CROSS_DELAY_MS);
    },LP_MS);
  },{passive:true});
  gd.addEventListener('touchmove',function(e){
    if(mode==='pinch'){
      if(e.touches.length===2&&pinch){
        e.preventDefault();
        var f=fracOf((e.touches[0].clientX+e.touches[1].clientX)/2);
        var ns=clampSpan(pinch.span*Math.pow(pinch.d/dist(e.touches),ZOOM_GAIN));
        span=ns; i0=clampI0(pinch.anchor-f*ns,ns); schedule();
      }
      return;
    }
    if(e.touches.length!==1)return;
    var t=e.touches[0]; lastT=t;
    if(mode==='arming'){e.preventDefault(); return;}
    if(mode==='cross'){e.preventDefault(); move(t.clientX,t.clientY); return;}
    if(mode==='pan'){e.preventDefault(); i0=clampI0(startI0-pxToBars(t.clientX-sx)*PAN_GAIN,span); schedule(); return;}
    if(mode==='scroll')return;
    var dx=t.clientX-sx, dy=t.clientY-sy;
    if(Math.abs(dx)>6||Math.abs(dy)>6){
      clearTimeout(timer);
      if(Math.abs(dx)>Math.abs(dy)){mode='pan'; e.preventDefault(); i0=clampI0(startI0-pxToBars(dx)*PAN_GAIN,span); schedule();}
      else mode='scroll';
    }
  },{passive:false});
  function touchEnd(e){
    clearTimeout(timer); lastTouch=Date.now();
    if(e.touches&&e.touches.length>0)return;      // 아직 손가락이 남아 있음
    pinch=null;
    if(mode===null&&e.type==='touchend'&&Date.now()-tStart<300){   // 짧게 톡 → 두 번이면 기본값
      var now=Date.now();
      if(now-lastTap<350){resetView(); lastTap=0;} else lastTap=now;
    }
    if(mode!=='cross')mode=null;                   // 십자선은 손을 떼도 남고, 다음 터치 때 사라짐
  }
  gd.addEventListener('touchend',touchEnd);
  gd.addEventListener('touchcancel',touchEnd);
  gd.addEventListener('contextmenu',function(e){e.preventDefault();});

  // ── PC: 마우스를 누른 채 끌면 이동, 그냥 움직이면 십자선, Ctrl+휠=확대/축소 ──
  function emu(e){return (Date.now()-lastTouch<1000)||(e.sourceCapabilities&&e.sourceCapabilities.firesTouchEvents);}
  gd.addEventListener('mousedown',function(e){if(emu(e))return; mdown=true; sx=e.clientX; startI0=i0; e.preventDefault();});
  window.addEventListener('mousemove',function(e){
    if(!mdown)return;
    i0=clampI0(startI0-pxToBars(e.clientX-sx),span); schedule();
  },true);
  window.addEventListener('mouseup',function(){mdown=false;},true);
  gd.addEventListener('mousemove',function(e){if(emu(e))return; move(e.clientX,e.clientY);});
  gd.addEventListener('mouseleave',function(e){if(emu(e))return; hide();});
  gd.addEventListener('dblclick',function(e){resetView();});
  gd.addEventListener('wheel',function(e){
    if(!e.ctrlKey)return; e.preventDefault();
    var f=fracOf(e.clientX), anchor=i0+f*span;
    var ns=clampSpan(span*(e.deltaY>0?1.23:0.81));
    if(ns===span)ns=clampSpan(span+(e.deltaY>0?1:-1));
    span=ns; i0=clampI0(anchor-f*ns,ns); schedule();
  },{passive:false});
  updateSummary();
  render();
}
"""


def finish(fig, low_label):
    m = len(dates)
    pos = sorted({m - 1 - int(round(i * (m - 1) / 3.6)) for i in range(4)})
    fig.update_xaxes(type="category", tickmode="array",
                     tickvals=[dates[i] for i in pos],
                     ticktext=[pd.to_datetime(dates[i]).strftime("%m.%d") for i in pos],
                     tickangle=0, tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#f0b8b8", gridwidth=1,
                     showline=True, mirror=True, linecolor="#9aa3ae", linewidth=1, zeroline=False)   # 차트 영역 테두리
    fig.update_xaxes(showticklabels=True, row=1, col=1)    # 날짜는 위 차트 아래(두 차트 사이)에 표시
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_yaxes(side="right", tickformat=",.0f", tickfont=dict(size=11, color="#555"),
                     showgrid=True, gridcolor="#eeeeee", zeroline=False, ticks="",
                     showline=True, mirror=True, linecolor="#9aa3ae", linewidth=1)
    fig.update_yaxes(nticks=4, row=2, col=1)
    tagc = "#e5484d" if chg > 0 else "#3b6fd4"
    # 현재가 태그(오른쪽 축)
    fig.add_annotation(xref="paper", x=1.0, xanchor="left", yref="y", y=yv[-1], showarrow=False,
                       text=f"{last:,.0f}<br>{pct:+.2f}%", align="left", borderpad=2,
                       font=dict(color="white", size=11), bgcolor=tagc)
    # 아래 차트 이름
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=fig.layout.yaxis2.domain[1],
                       xanchor="left", yanchor="top", showarrow=False,
                       text=low_label, font=dict(size=11, color="#555"))
    fig.update_layout(height=620, margin=dict(l=4, r=max(66, 12 + 7 * len(f"{last:,.0f}")), t=26, b=22),
                      showlegend=False, bargap=0.05, hovermode=False, dragmode=False,
                      xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def show(fig, div_id, bars):
    """좌우 이동 + 길게 눌러 십자선 기능을 얹어 차트를 표시"""
    def pack(v):
        return np.round(np.asarray(v, dtype=float), 2).tolist()
    series = [pack(v) for v in (yv, curvev, upv, lov)]
    dump = lambda o: json.dumps(o, separators=(",", ":"))
    idx_json = dump(pack(idx_scaled)) if idx_scaled is not None else "null"
    raw_json = dump(pack(idx_arr)) if idx_arr is not None else "null"
    open_json = dump(pack(ov))
    low_json = dump(pack(lv))
    idx2_json = dump(pack(idx_scaled_p)) if idx_scaled_p is not None else "null"
    post = "installCrosshair(document.getElementById('{plot_id}'), %d, %s, %s, %s, %d, %s, %s, %s, %s, %d, %s, %d, %s);" % (
        init_span, dump(series), dump(pack(bars)),
        idx_json, (4 if idx_scaled is not None else -1), json.dumps(idx_name),
        raw_json, open_json, low_json, n - init_span, idx2_json, 5, dump(pack(hv)))
    html = "<script>" + CROSSHAIR_JS + "</script>" + fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id=div_id,
        config={"staticPlot": True, "responsive": True},
        post_script=post)
    components.html(html, height=640)


t_calc = time.time() - _tc
_td = time.time()
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
base_traces(fig)
cv = curvev - curvev.min()
fig.add_trace(go.Bar(x=dates, y=cv, marker_color=direction_colors(curvev),
                     name="곡선추세 방향", showlegend=False), row=2, col=1)
show(finish(fig, "곡선추세 방향"), "chart_trend", cv)
status.empty()
top_ph.markdown(top_bar(timing_html([("시세", t_fetch), ("계산", t_calc), ("차트", time.time() - _td)])),
                unsafe_allow_html=True)
if idx_arr is None:
    st.caption("※ 지수 데이터를 불러오지 못해 비교선 없이 표시 중입니다.")
elif idx_note:
    st.caption(idx_note)
